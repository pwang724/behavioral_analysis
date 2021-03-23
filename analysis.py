import matplotlib.pyplot as plt
import numpy as np
import glob
import imageio
import itertools
import pickle
import pandas as pd
import seaborn as sns
import anipose_scripts.anipose_main as main
import anipose_scripts.constants as constants
import tools
import pprint
import itertools
import os

plt.rcParams.update({'font.size': 15})

FRAME_RATE = 40
NAN_GAP = 3
SCORE_THRESHOLD = 0.9
SCORE_INTERP = 2
CRITERIA_BP_FRACTION = 0.75
CRITERIA_CONTIGUOUS_SECONDS = .2
CRITERIA_CONTIGUOUS_FRAMES = int(np.ceil(CRITERIA_CONTIGUOUS_SECONDS *
                                         FRAME_RATE))
INTERP_SMOOTH_MOTION_COEFFICIENT = 30
DIM_CAMERA = 0
DIM_BP = 1
DIM_FRAME = 2
DIM_COORDS_SCORE = 3
BPS_TRACK_LOCATION = ['r2_in', 'r3_in']
BPS_TRACK_PAW = ['r1_in', 'r1_out',
                 'r2_in', 'r2_out',
                 'r3_in', 'r3_out',
                 'r4_in', 'r4_out']
BPS_TRACK_PELLET = ['pellet']
BPS_CARE = {'location': BPS_TRACK_LOCATION,
            'paw': BPS_TRACK_PAW,
            'pellet': BPS_TRACK_PELLET}
LABEL_MAX_HOLE_LENGTH = 3
LABEL_MIN_REGION_LENGTH = 4

ANGLES_LOCATION = [
    ['r2_out', 'r2_in', 'r3_in'],
    ['r3_out', 'r3_in', 'r2_in'],
]
ANGLES_PAW = [
    ['r1_out', 'r1_in', 'r2_in'],
    ['r2_out', 'r2_in', 'r3_in'],
    ['r3_out', 'r3_in', 'r4_in'],
    ['r2_out', 'r2_in', 'r1_in'],
    ['r3_out', 'r3_in', 'r2_in'],
    ['r4_out', 'r4_in', 'r3_in']
]
BPS_ANGLES = {'location': ANGLES_LOCATION,
              'paw': ANGLES_PAW,
              'pellet': []}


def find_region_bounds(bool_array, min_gap=0, max_gap=0):
    assert (bool_array.dtype == 'bool')
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


def repair_holes(x, y, score, score_threshold, default_interp_score_val,
                 gap):
    failed_threshold = score < score_threshold
    regions, interp_ixs = find_region_bounds(failed_threshold, max_gap=gap)
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

    score[interp_ixs] = default_interp_score_val
    return x, y, score


def load_to_np_array(file_name, bps):
    df = pd.read_hdf(file_name)
    scorer = df.columns.levels[0][0]
    df = df.loc[:,scorer]

    n_frames = df.shape[0]
    n_joints = len(bps)
    assert len(df.index) == n_frames
    data = np.array(df[bps]).reshape(n_frames, n_joints, 3)
    data = np.transpose(data, (1, 0, 2))
    return data

def get_good_chunks(data_interp,
                    bps_ix,
                    score_threshold,
                    min_frames):
    relevant_scores = data_interp[bps_ix,:,2]
    bool_arr = relevant_scores > score_threshold
    bool_vec = np.all(bool_arr, axis=0)
    regions, region_ixs = find_region_bounds(bool_vec, min_gap=min_frames)
    return regions, region_ixs

# inputs
data_dir = r'C:\Users\Peter\Desktop\DATA\M4\2021_03_10'
analyzed_dir = os.path.join(data_dir, 'ANALYZED')
scheme = [['r1_in', 'r1_out'],
          ['r2_in', 'r2_out'],
          ['r3_in', 'r3_out'],
          ['r4_in', 'r4_out'],
          ['r1_in', 'r2_in', 'r3_in', 'r4_in'],
          ['pellet'],
          ['insured pellet']]

# load markers
marker_dir = os.path.join(analyzed_dir, 'POSE_2D')
list_of_marker_files = [tools.get_files(marker_dir, (x + '.h5'))
                        for x in constants.CAMERA_NAMES]
list_of_marker_files = [x for x in zip(*list_of_marker_files)]
bps_to_include = np.unique([x for y in scheme for x in y])
bp_dict = {bp: i for i, bp in enumerate(bps_to_include)}

# marker preprocessing
marker_xys_per_video = [] #[video][camera][data]
marker_regions_per_video = [] #[video][camera]['location/paw/pellet'][regions]
for marker_file_per_camera in list_of_marker_files:
    marker_xys = []
    marker_regions = []
    for fn in marker_file_per_camera:
        # DATA SHAPE [BODYPARTS, FRAMES, (X, Y, SCORE)]
        data = load_to_np_array(fn, bps_to_include)
        for bpix in range(data.shape[0]):
            x, y, score = repair_holes(x=data[bpix, :, 0],
                                       y=data[bpix, :, 1],
                                       score=data[bpix, :, 2],
                                       score_threshold=SCORE_THRESHOLD,
                                       default_interp_score_val=SCORE_INTERP,
                                       gap=NAN_GAP)
            data[bpix, :, 0] = x
            data[bpix, :, 1] = y
            data[bpix, :, 2] = score
        marker_xys.append(data)

        bps_regions_dict = {}
        for k, bps in BPS_CARE.items():
            bps_ix = [bp_dict[bp] for bp in bps]
            regions, region_ixs = get_good_chunks(data,
                                                  bps_ix,
                                                  score_threshold=SCORE_THRESHOLD,
                                                  min_frames=CRITERIA_CONTIGUOUS_FRAMES)
            bps_regions_dict[k] = regions
        marker_regions.append(bps_regions_dict)
    marker_xys_per_video.append(marker_xys)
    marker_regions_per_video.append(marker_regions)

# load labels
label_dir = os.path.join(analyzed_dir, 'LABELS')
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
        _, repair_ixs = find_region_bounds(failed_threshold,
                                           max_gap=LABEL_MAX_HOLE_LENGTH)
        label_vec[repair_ixs] = 1
        chunks, _ = find_region_bounds(label_vec > 0,
                                       min_gap=LABEL_MIN_REGION_LENGTH)
        label_regions[label_name] = chunks
    label_regions_per_video.append(label_regions)

# get grab locations
bps = BPS_CARE['location']
bps_ix = [bp_dict[bp] for bp in bps]
grab_xy_per_video = []
grab_ixs_per_video = []
for marker_xys, label_regions in zip(marker_xys_per_video,
                                     label_regions_per_video):
    _, loc_ixs = get_good_chunks(marker_xys[1],
                                 bps_ix,
                                 score_threshold=SCORE_THRESHOLD,
                                 min_frames=CRITERIA_CONTIGUOUS_FRAMES)
    xcam0 = np.mean(marker_xys[0][bps_ix, :, 0], axis=0)
    xcam1 = np.mean(marker_xys[1][bps_ix, :, 0], axis=0)
    grab_regions = label_regions['grab']
    grab_xy = []
    grab_ix = []
    for r in grab_regions:
        grab_ixs = np.arange(r[0], r[1])
        ixs = np.intersect1d(grab_ixs, loc_ixs)
        x0 = xcam0[ixs]
        x1 = xcam1[ixs]
        xy = [np.mean(x0[:3]), np.mean(x1[:3])]
        grab_xy.append(xy)
        grab_ix.append(list(ixs[:3]))
    grab_xy_per_video.append(grab_xy)
    grab_ixs_per_video.append(grab_ix)

# pellet
SCORE_THRESHOLD = 0.99
bp_ixs = [bp_dict[bp] for bp in BPS_TRACK_PELLET]
data = marker_xys_per_video[0][1]

regions, region_ixs = get_good_chunks(data,
                                      bps_ix,
                                      score_threshold=SCORE_THRESHOLD,
                                      min_frames=50)


