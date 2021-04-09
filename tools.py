import os
import numpy as np
import pandas as pd
import re

def get_files(source, wcs):
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith(wcs):
                matches.append(os.path.join(root, filename))
    return matches


def print_dict(ddict, length=True):
    for k, v in ddict.items():
        if length:
            print(f'{k}: {len(v)}')
        else:
            print(f'{k}: {v}')


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', "
                          "'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    y = y[(window_len//2):-(window_len//2)]
    assert len(x) == len(y)
    return y


def load_pose_to_np(fname, bps):
    '''
    Load DLC pose2d data into an np array with specified body parts.

    :param fname: DLC hd5 file
    :param bps: list of body part names
    :return: numpy array of dims: [bps, frames, xys]
    '''
    df_orig = pd.read_hdf(fname)
    scorer = df_orig.columns.levels[0][0]
    df = df_orig.loc[:,scorer]

    n_frames = df.shape[0]
    n_joints = len(bps)
    assert len(df.index) == n_frames
    data = np.array(df[bps]).reshape(n_frames, n_joints, 3)
    data = np.transpose(data, (1, 0, 2))
    return data


def load_pose_2d(fname):
    df_orig = pd.read_hdf(fname)
    scorer = df_orig.columns.levels[0][0]
    data = df_orig.loc[:, scorer]

    bp_index = data.columns.names.index('bodyparts')
    coord_index = data.columns.names.index('coords')
    bodyparts = list(data.columns.get_level_values(bp_index).unique())
    n_possible = len(data.columns.levels[coord_index])//3

    n_frames = len(data)
    n_joints = len(bodyparts)
    out = np.array(data).reshape(n_frames, n_joints, n_possible, 3)

    metadata = {
        'bodyparts': bodyparts,
        'scorer': scorer,
        'index': data.index
    }
    return out, metadata


def write_pose_2d(all_points, metadata, outname=None):
    points = all_points[:, :, :2]
    scores = all_points[:, :, 2]

    scorer = metadata['scorer']
    bodyparts = metadata['bodyparts']
    index = metadata['index']

    columns = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'])

    dout = pd.DataFrame(columns=columns, index=index)

    dout.loc[:, (scorer, bodyparts, 'x')] = points[:, :, 0]
    dout.loc[:, (scorer, bodyparts, 'y')] = points[:, :, 1]
    dout.loc[:, (scorer, bodyparts, 'likelihood')] = scores

    if outname is not None:
        dout.to_hdf(outname, 'df_with_missing', format='table', mode='w')

    return dout


def true_basename(fname):
    basename = os.path.basename(fname)
    basename = os.path.splitext(basename)[0]
    return basename


def camname_from_regex(cam_regex, fname):
    basename = true_basename(fname)
    match = re.search(cam_regex, basename)
    if not match:
        return None
    else:
        name = match.groups()[0]
        return name.strip()


def videoname_from_regex(cam_regex, fname):
    basename = true_basename(fname)
    vidname = re.sub(cam_regex, '', basename)
    return vidname.strip()


def desktop_path():
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    return desktop