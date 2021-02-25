import cv2
import numpy as np
import glob
import yaml
import os
import camera_calibration.tools as tools
import camera_calibration.constants as c

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((c.NUM_SQUARES, 3), np.float32)
objp[:, :2] = np.mgrid[0:c.CHECKER_ROWS, 0:c.CHECKER_COLS].T.reshape(-1, 2)
objp *= c.GRID_SPACING_MM

xyz = np.float32([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]).reshape(-1, 3)
xyz *= c.GRID_SPACING_MM

yaml_path = r'..\camera_calibration\outputs\intrinsic_calibration.yaml'
with open(yaml_path) as file:
    documents = yaml.full_load(file)
lmat = documents['lmat']
lmat = np.float32(lmat)
ldist = documents['ldist']
ldist = np.float32(ldist)


# Load data
image_path = c.IM_DIR
bad_ixs = [str(x).zfill(2) for x in np.arange(11, 20)]
image_dict = {}
for image_wc in [c.CAM_ONE_ORIENT, c.CAM_TWO_ORIENT]:
    images = [im for im in os.listdir(c.IM_DIR) if c.IM_EXTENSION in im and
              image_wc in im]
    bad_images = [image_wc + ix + '.' + c.IM_EXTENSION for ix in bad_ixs]
    good_images = sorted(list(set(images) - set(bad_images)))
    image_dict[image_wc] = [os.path.join(c.IM_DIR, x) for x in good_images]

objp, lpt, rpt, wh = tools.get_stereo_points(image_dict[c.CAM_ONE_ORIENT],
                                             image_dict[c.CAM_TWO_ORIENT],
                                             draw=False)

mat = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float32)
xyz = np.matmul(mat, xyz)

for fname in image_dict[c.CAM_ONE_ORIENT]:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (c.CHECKER_ROWS, c.CHECKER_COLS), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,
                                    corners,
                                    c.WINDOW_SIZE,
                                    (-1, -1),
                                    criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(
            objp[0],
            corners2,
            lmat,
            ldist)



        # project 3D points to image plane

        imgpts, jac = cv2.projectPoints(xyz, rvecs, tvecs, lmat, ldist)

        img = tools.draw(img, corners2, imgpts)
        cv2.imshow(os.path.split(fname)[1], img)
        k = cv2.waitKey(0) & 0xff
        if k == 's':
            cv2.imwrite(fname[:6] + '.png', img)

# cv2.destroyAllWindows()
