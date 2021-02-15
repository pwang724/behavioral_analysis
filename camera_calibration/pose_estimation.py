import cv2
import numpy as np
import glob
import yaml
import os

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

rows = 6
cols = 9
squares = rows * cols
objp = np.zeros((squares,3), np.float32)
objp[:,:2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

yaml_path = r'..\camera_calibration\calibration_matrix.yaml'
with open(yaml_path) as file:
    documents = yaml.full_load(file)
mtx = documents['camera_matrix']
mtx = np.float32(mtx)
dist = documents['dist_coeff']
dist = np.float32(dist)

image_path = r'../images/1/'
image_wc = 'left'

images = glob.glob(image_path + image_wc + '*.jpg')
filtered_images = []
bad = [
    'left12.jpg',
    'left13.jpg',
    'left14.jpg',
    'left15.jpg',
    'left16.jpg',
    'left17.jpg',
    'left18.jpg',
    'left19.jpg']
for im in images:
    add = True
    for b in bad:
        if b in im:
            add = False
    if add:
        filtered_images.append(im)

for fname in filtered_images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp,
                                                      corners2,
                                                      mtx,
                                                      dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img,corners2,imgpts)
        cv2.imshow(os.path.split(fname)[1], img)
        k = cv2.waitKey(0) & 0xff
        if k == 's':
            cv2.imwrite(fname[:6]+'.png', img)

# cv2.destroyAllWindows()

