import cv2
import numpy as np
import glob
import yaml
import os
import camera_calibration.tools as tools
import camera_calibration.constants as c

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

rows = 6
cols = 9
squares = rows * cols
objp = np.zeros((squares, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
xyz = np.float32([
    [3, 0, 0],
    [0, 3, 0],
    [0, 0, 3]
]).reshape(-1, 3)

yaml_path = r'..\camera_calibration\outputs\intrinsic_calibration.yaml'
with open(yaml_path) as file:
    documents = yaml.full_load(file)
lmat = documents['lmat']
lmat = np.float32(lmat)
ldist = documents['ldist']
ldist = np.float32(ldist)


# Load data
image_path = r'../images/1/'
bad_ixs = [str(x).zfill(2) for x in np.arange(11, 20)]
image_dict = {}
for image_wc in ['left', 'right']:
    images = [im for im in os.listdir(c.IMAGE_DIR) if c.IM_EXTENSION in im and
              image_wc in im]
    bad_images = [image_wc + ix + '.jpg' for ix in bad_ixs]
    good_images = sorted(list(set(images) - set(bad_images)))
    image_dict[image_wc] = [os.path.join(c.IMAGE_DIR, x) for x in good_images]

objp, lpt, rpt, wh = tools.get_stereo_points(image_dict['left'],
                                             image_dict['right'],
                                             draw=False)

for fname in image_dict['left']:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

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
