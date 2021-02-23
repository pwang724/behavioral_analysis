import cv2

INTRINSIC_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                  30,
                  1e-4)
STEREO_CRIT = (
    cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
    30,
    1e-4)

STEREO_FLAGS = cv2.CALIB_FIX_INTRINSIC

CHECKER_ROWS = 6
CHECKER_COLS = 9
NUM_SQUARES = CHECKER_ROWS * CHECKER_COLS

WINDOW_SIZE = (11, 11)

OUTPUT_FOLDER_NAME = 'outputs'
IM_EXTENSION = 'jpg'
IMAGE_DIR = r'../images/1/'