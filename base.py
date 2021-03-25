from enum import Enum


class OUTCOME(Enum):
    SUCCESS = 'True'
    FAIL = 'False'
    NO_PELLET = 'No pellet'
    GROOM = 'Groom'
    NO_ATTEMPTS = 'No attempts'
    DROP_AFTER_GRAB = 'Dropped after grab'
    DROP_FP = 'Drop FP'
    DROP_FN = 'Drop FN'


class GRABTYPES(Enum):
    DROPPED = 'Dropped'
    SNATCHED = 'Snatched'
    FAIL_WITH_PELLET = 'Failed with pellet present'
    FAIL_POST_DROP = 'Failed post drop'
    FAIL_POST_SNATCH = 'Failed post snatch'


OUTCOME_TO_KEY_DICT = {
    'T': OUTCOME.SUCCESS,
    'F': OUTCOME.FAIL,
    'D': OUTCOME.DROP_AFTER_GRAB,
    'G': OUTCOME.GROOM,
    'X': OUTCOME.NO_PELLET,
    'L': OUTCOME.FAIL,
}

# Frame rate of video
FRAME_RATE = 40
# Extension of collected videos
RAW_VIDEO_EXT = 'avi'
# Camera names
CAMERA_NAMES = ['CAM0', 'CAM1']
# Regex to search for relevant files based on camera names
CAM_REGEX = 'CAM([0-9])'
# Calibration board
# CALIBRATION_BOARD = Checkerboard(squaresX=9,
#                                  squaresY=6,
#                                  square_length=1,
#                                  manually_verify=False)

## Marker configs

# Score threshold for DLC marker confidence
SCORE_THRESHOLD = 0.9
# Maximum frame length for interpolating DLC markers with scores below score threshold
REPAIR_FRAME_GAP = 3
# Placeholder score value for frames in which marker locations were interpolated
SCORE_INTERP = 2
# Minimum length for a contiguous chunk of DLC marker locations
CRITERIA_CONTIGUOUS_FRAMES = 8
# Interpolation coefficient for spline-based smooothing of reach trajectory
INTERP_SMOOTH_MOTION_COEFFICIENT = 30

## Label configs

# Maximum frame length for joining separated regions
LABEL_JOIN_FRAME_GAP = 3
# Minimum length for a contiguous chunk of labels
LABEL_CRITERIA_CONTIGUOUS_FRAMES = 4

## Anipose configs
