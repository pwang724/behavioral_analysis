import itertools
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import base
import tools


def _get_markers(marker_dir, bodyparts, cams):
    '''
    Returns data in the form of [trial][camera][body parts, frames, xys]

    :param marker_dir:
    :param bodyparts:
    :param cams:
    :return:
    '''
    marker_files = [tools.get_files(marker_dir, (x + '.h5')) for x in cams]
    marker_files = [x for x in zip(*marker_files)]
    marker_xys_per_video = [] #[video][camera][data]
    for marker_file_per_camera in marker_files:
        marker_xys = []
        for fn in marker_file_per_camera:
            data = tools.load_pose_to_np(fn, bodyparts)
            for bpix in range(data.shape[0]):
                x, y, score = _repair_bodyparts(
                    x=data[bpix, :, 0],
                    y=data[bpix, :, 1],
                    score=data[bpix, :, 2])
                data[bpix, :, 0] = x
                data[bpix, :, 1] = y
                data[bpix, :, 2] = score
            marker_xys.append(data)
        marker_xys_per_video.append(marker_xys)
    return marker_xys_per_video


def _get_labels(label_dir):
    '''
    Returns data in the form of a list of dictionaries with
    key: bp
    value: list of regions of form [start, end].

    Also returns label names.

    :param label_dir:
    :return:
    '''
    list_of_labels = tools.get_files(label_dir, ('labels.csv'))
    label_names = pd.read_csv(list_of_labels[0], index_col=0).columns.to_numpy()

    # label preprocessing
    label_regions_per_video = []
    for fn in list_of_labels:
        df = pd.read_csv(fn, index_col=0)
        df = df.to_numpy().T
        label_regions = {}
        for label_vec, label_name in zip(df, label_names):
            failed_threshold = label_vec < 1
            _, repair_ixs = _chunk(
                failed_threshold,
                max_gap=base.LABEL_JOIN_FRAME_GAP)
            label_vec[repair_ixs] = 1
            chunks, _ = _chunk(
                label_vec > 0,
                min_gap=base.LABEL_CRITERIA_CONTIGUOUS_FRAMES)
            label_regions[label_name] = chunks
        label_regions_per_video.append(label_regions)
    return label_regions_per_video, label_names


def _chunk(bool_array, min_gap=0, max_gap=0):
    idx = np.diff(np.r_[0, bool_array, 0]).nonzero()[0]
    regions = np.reshape(idx, (-1, 2))
    if min_gap > 0 and max_gap > 0:
        regions = np.array([r for r in regions if (
                (r[1] - r[0]) <= max_gap and (r[1] - r[0] >= min_gap))])
    elif min_gap > 0 and max_gap == 0:
        regions = np.array([r for r in regions if (r[1] - r[0]) >= min_gap])
    elif min_gap == 0 and max_gap > 0:
        regions = np.array([r for r in regions if (r[1] - r[0]) <= max_gap])
    else:
        raise ValueError('both min and max gaps are 0')
    region_ixs = np.array(
        list(itertools.chain(*[range(r[0], r[1]) for r in regions]))).astype(
        int)
    return regions, region_ixs


def _repair_bodyparts(x, y, score):
    failed_threshold = score < base.SCORE_THRESHOLD
    regions, interp_ixs = _chunk(failed_threshold, max_gap=base.REPAIR_FRAME_GAP)
    good_ix = np.arange(len(x))
    good_ix = np.delete(good_ix, interp_ixs)
    good_x = np.delete(x, interp_ixs)
    new_x = np.interp(interp_ixs, good_ix, good_x)
    x[interp_ixs] = new_x

    good_ix = np.arange(len(y))
    good_ix = np.delete(good_ix, interp_ixs)
    good_y = np.delete(y, interp_ixs)
    new_y = np.interp(interp_ixs, good_ix, good_y)
    y[interp_ixs] = new_y

    score[interp_ixs] = base.SCORE_INTERP
    return x, y, score


def _chunk_bodyparts(scores,
                     score_threshold,
                     min_frames):
    '''
    :param scores: shape [bps, frames]
    :param score_threshold:
    :param min_frames:
    :return:
    '''
    bool_arr = scores > score_threshold
    bool_vec = np.all(bool_arr, axis=0)
    regions, region_ixs = _chunk(bool_vec, min_gap=min_frames)
    return regions, region_ixs


def get_pellet_location(loc_arr, score_arr, plot):
    '''
    1. Assumes that xs and xss are of shape [TRIALS, FRAMES]
    2. Assumes that the wheel rotation speed did not change within a session

    :param loc_arr:
    :param score_arr:
    :return:
    '''
    import scipy.ndimage
    num_start_frames = 100
    burn_in_frames = 5
    stationary_win = 5
    score_threshold = base.SCORE_THRESHOLD

    # find the best first stationary index
    loc_arr = loc_arr[:, :num_start_frames]
    score_arr = score_arr[:, :num_start_frames]
    xmas = np.ma.masked_where(score_arr < score_threshold, loc_arr)
    dxs = np.diff(xmas, axis=1)
    mdxs = np.mean(dxs, axis=0)
    ix = np.argwhere(mdxs[burn_in_frames:] > 0)[0][0] + burn_in_frames

    # find all coordinates in which the pellet at the stationary index is
    # visible and has small velocity
    mask = np.logical_and(score_arr[:, ix:ix + stationary_win] > score_threshold,
                          np.abs(dxs[:, ix:ix + stationary_win]) < 0.2)
    coords = np.ma.masked_where(np.invert(mask), loc_arr[:, ix:ix + stationary_win])
    mcoords = np.mean(coords, axis=1)

    # interpolate and filter these coordinates across trials
    mcoords_interp = np.copy(mcoords)
    mask = np.ma.getmask(mcoords)
    interp_ix = np.where(mask)[0]
    known_ix = np.where(np.invert(mask))[0]
    interp_vals = np.interp(interp_ix, known_ix, mcoords[known_ix])
    mcoords_interp[interp_ix] = interp_vals
    mcoords_interp = scipy.ndimage.median_filter(mcoords_interp, size=5)

    if plot:
        plt.figure()
        plt.plot(mdxs)
        plt.title('Velocity')
        plt.figure()
        plt.plot(mcoords)
        plt.title('Unrefined coordinates')
        plt.figure()
        plt.plot(mcoords_interp)
        plt.title('Interpolated + Median filtered coordinates')
    return mcoords_interp


def _get_grab_locations_atomic(marker_positions,
                               marker_scores,
                               grab_region):
    '''
    Find grab endpoints with viable paw locations. Assumes that the first
    3 frames of grab_region contains the grab endpoint.

    1. Assumes marker_positions has shape [body parts, frames]
    2. Assumes marker_scores has shape [body parts, frames]
    3. Assumes grab_region is of shape [start_ix, end_ix]

    :param marker_positions:
    :param label_regions:
    :return:
    '''
    score_threshold = base.SCORE_THRESHOLD
    min_contiguous_frames = base.CRITERIA_CONTIGUOUS_FRAMES
    window = 3

    _, loc_ixs = _chunk_bodyparts(marker_scores,
                                  score_threshold=score_threshold,
                                  min_frames=min_contiguous_frames)
    grab_ixs = np.arange(grab_region[0], grab_region[1])
    ixs = np.intersect1d(grab_ixs, loc_ixs)
    mean_marker_position = np.mean(marker_positions, axis=0)
    pos = np.mean(mean_marker_position[ixs][:window])
    return pos


def outcome_diagnostics(annotated_outcomes: List[base.OUTCOME],
                        computed_outcomes: List[base.OUTCOME],
                        mask):
    '''
    Compare ground_truth labels with computed outcomes. Print mistakes if
    there are discrepancies that are not masked already
    :param mask: bool array of same length as each outcome
    '''
    assert len(annotated_outcomes) == len(computed_outcomes)
    assert len(annotated_outcomes) == len(mask)
    for i, (x, y) in enumerate(zip(annotated_outcomes, computed_outcomes)):
        if x!=y and mask[i]:
            print(i, x, y, 'MASKED' if not mask[i] else '')


def outcome_truth_table(dropped_regions, chew_regions, grabbed_regions):
    '''
    All inputs are list of regions. Can be empty.
    :param dropped_regions:
    :param chew_regions:
    :param grabbed_regions:
    :return:
    '''
    any_drop = len(dropped_regions) > 0
    any_chew = len(chew_regions) > 0
    any_grab = len(grabbed_regions) > 0
    if not any_drop and any_chew:
        outcome = base.OUTCOME.SUCCESS
    elif any_drop and not any_chew:
        outcome = base.OUTCOME.FAIL
    elif any_drop and any_chew:
        if chew_regions[0][0] <= dropped_regions[0][0]:
            outcome = base.OUTCOME.DROP_AFTER_GRAB
        else:
            outcome = base.OUTCOME.DROP_FP
    elif len(dropped_regions) == 0 and len(chew_regions) == 0:
        if any_grab:
            outcome = base.OUTCOME.FAIL
        else:
            outcome = base.OUTCOME.NO_ATTEMPTS
    else:
        raise ValueError('outcome is not recognized')
    return outcome


def grab_truth_table(outcome, grabbed_regions, chew_regions, dropped_regions):
    assert len(dropped_regions) <= 1, i

    done = False
    pregrab = base.GRABTYPES.FAIL_WITH_PELLET
    grab_outcomes = []
    if outcome == base.OUTCOME.FAIL and len(dropped_regions) == 0:
        grab_outcomes = [pregrab for x in grabbed_regions]
        done = True
    elif outcome == base.OUTCOME.FAIL and len(dropped_regions) > 0:
        during = base.GRABTYPES.DROPPED
        post = base.GRABTYPES.FAIL_POST_DROP
        key_region = dropped_regions[0]
    elif outcome == base.OUTCOME.DROP_AFTER_GRAB:
        during = base.GRABTYPES.DROPPED
        post = base.GRABTYPES.FAIL_POST_DROP
        key_region = chew_regions[0]
    elif outcome == base.OUTCOME.SUCCESS or outcome == base.OUTCOME.DROP_FP:
        during = base.GRABTYPES.SNATCHED
        post = base.GRABTYPES.FAIL_POST_SNATCH
        key_region = chew_regions[0]
    elif outcome == base.OUTCOME.NO_ATTEMPTS:
        grab_outcomes = []
        done = True
    else:
        raise ValueError(f'outcome: {outcome} not recognized')

    if not done:
        n_grabs = len(grabbed_regions)
        for g in range(n_grabs):
            current_grab = grabbed_regions[g]
            if current_grab[1] <= key_region[0]:
                if g + 1 < n_grabs:
                    next_grab = grabbed_regions[g+1]
                else:
                    next_grab = None

                if next_grab is None or next_grab[0] >= key_region[1]:
                    grab_outcomes.append(during)
                    assert current_grab[1] - key_region[0] < 100
                elif next_grab[0] < key_region[1]:
                    grab_outcomes.append(base.GRABTYPES.FAIL_WITH_PELLET)
                else:
                    raise ValueError('possibility not considered')
            else:
                grab_outcomes.append(post)
    return grab_outcomes