from aniposelib.boards import Checkerboard


VIDEO_EXT = 'avi'
DLC_LIKELIHOOD_THRES = 0.9
FILTER_CONFIG = {
    'enabled': False,
    'type': 'medfilt',
    'medfilt': 13,
    'offset_threshold': 25,
    'score_threshold': DLC_LIKELIHOOD_THRES,
    'spline': True,
    'n_back': 5,
    'multiprocessing': False
}
FILTER3D_CONFIG = {
    'error_threshold': 50,
    'score_threshold': 0.95,
    'medfilt': 13
}
CAMERA_NAMES = ['CAM0', 'CAM1']
CAM_REGEX = 'CAM([0-9])'
CALIBRATION_BOARD = Checkerboard(squaresX=9,
                                 squaresY=6,
                                 square_length=1,
                                 manually_verify=False)