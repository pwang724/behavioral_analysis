import numpy as np
import cv2
import camera_calibration.constants as c
import os

def get_stereo_points(left_images, right_images, draw):
    objpoints = []  # 3d point in real world space
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((c.NUM_SQUARES, 3), np.float32)
    objp[:, :2] = np.mgrid[0:c.CHECKER_ROWS, 0:c.CHECKER_COLS].T.reshape(-1, 2)

    for left_name, right_name in zip(left_images, right_images):
        left_img = cv2.imread(left_name)
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        left_ret, left_corners = cv2.findChessboardCorners(
            left_gray, (c.CHECKER_ROWS, c.CHECKER_COLS), None)

        right_img = cv2.imread(right_name)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        right_ret, right_corners = cv2.findChessboardCorners(
            right_gray, (c.CHECKER_ROWS, c.CHECKER_COLS), None)

        if left_ret and right_ret:
            objpoints.append(objp)
            left_corners2 = cv2.cornerSubPix(left_gray,
                                             left_corners,
                                             c.WINDOW_SIZE,
                                             (-1, -1),
                                             c.INTRINSIC_CRIT)
            imgpoints_left.append(left_corners2)

            right_corners2 = cv2.cornerSubPix(right_gray,
                                              right_corners,
                                              c.WINDOW_SIZE,
                                              (-1, -1),
                                              c.INTRINSIC_CRIT)
            imgpoints_right.append(right_corners2)

            if draw:
                left_img_cv = cv2.drawChessboardCorners(left_img,
                                                        (c.CHECKER_ROWS,
                                                         c.CHECKER_COLS),
                                                        left_corners2,
                                                        left_ret)
                cv2.imshow(os.path.split(left_name)[1], left_img_cv)

                right_img_cv = cv2.drawChessboardCorners(right_img,
                                                         (c.CHECKER_ROWS,
                                                          c.CHECKER_COLS),
                                                         right_corners2,
                                                         right_ret)
                cv2.imshow(os.path.split(right_name)[1], right_img_cv)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    return objpoints, imgpoints_left, imgpoints_right, \
           right_gray.shape[::-1]


def compute_projection_matrix(camera_matrix, rvec, tvec, rodrigues):
    if rodrigues:
        rvec = cv2.Rodrigues(rvec)[0]
    else:
        rvec = rvec
    RT = np.column_stack((rvec, tvec))
    return np.matmul(camera_matrix, RT)


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 5)
    return img