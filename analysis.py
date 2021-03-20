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
BPS_TRACK_LOCATION = ['r2_in', 'r2_out', 'r3_in', 'r3_out']
BPS_TRACK_PAW = ['r1_in', 'r1_out',
                 'r2_in', 'r2_out',
                 'r3_in', 'r3_out',
                 'r4_in', 'r4_out']
BPS_TRACK_PELLET = ['pellet']
BPS_CARE = {'location': BPS_TRACK_LOCATION,
            'paw': BPS_TRACK_PAW,
            'pellet': BPS_TRACK_PELLET}

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

dict_of_data = {x: tools.get_files(main.pose2d_folder, (x + '.h5'))
                for x in constants.CAMERA_NAMES}
bps_to_include = np.unique([x for y in main.scheme for x in y])
bp_dict = {bp: i for i, bp in enumerate(bps_to_include)}

### LOOP OVER THIS
video_ix = 0
cam = constants.CAMERA_NAMES[0]

# DATA SHAPE [BODYPARTS, FRAMES, (X, Y, SCORE)]
data = load_to_np_array(dict_of_data[cam][video_ix], bps_to_include)

# interpolate small missing regions per body part
for bp_ix in range(data.shape[0]):
    x, y, score = repair_holes(x=data[bp_ix, :, 0],
                               y=data[bp_ix, :, 1],
                               score=data[bp_ix, :, 2],
                               score_threshold=SCORE_THRESHOLD,
                               default_interp_score_val=SCORE_INTERP,
                               gap=NAN_GAP)
    data[bp_ix, :, 0] = x
    data[bp_ix, :, 1] = y
    data[bp_ix, :, 2] = score

# chunk into good segments
region_dict = {}
for k, bps in BPS_CARE.items():
    bps_ix = [bp_dict[bp] for bp in bps]
    regions, region_ixs = get_good_chunks(data,
                                          bps_ix,
                                          score_threshold=SCORE_THRESHOLD,
                                          min_frames=CRITERIA_CONTIGUOUS_FRAMES)
    region_dict[k] = regions
print(region_dict)

#


