import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import yaml

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

rows = 6
cols = 9
squares = rows * cols

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((squares,3), np.float32)
objp[:,:2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

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
print(np.array(filtered_images))

for image_name in filtered_images:
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (rows, cols), corners2, ret)
        # plt.imshow(img)
        # plt.title(os.path.split(image_name)[1])
        # plt.pause(1)

rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,None)

print("\nRMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())

# transform the matrix and distortion coefficients to writable lists
data = {'camera_matrix': np.asarray(camera_matrix).tolist(),
        'dist_coeff': np.asarray(dist_coefs).tolist()}

# and save it to a file
with open("calibration_matrix.yaml", "w") as f:
    yaml.dump(data, f)