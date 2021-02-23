import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import yaml

intrinsic_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                      30,
                      1e-4)
stereo_criteria = (
    cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
    30,
    1e-4)
stereo_flags = cv2.CALIB_FIX_INTRINSIC


rows = 6
cols = 9
squares = rows * cols
image_path = r'../images/1/'
bad = [str(x) for x in np.arange(11, 20)]

def compute_projection_matrix(camera_matrix, rvec, tvec, rodrigues):
    if rodrigues:
        rvec = cv2.Rodrigues(rvec)[0]
    else:
        rvec = rvec
    RT = np.column_stack((rvec, tvec))
    return np.matmul(camera_matrix, RT)


def stereo_retrieve(left_images, right_images, draw):
    objpoints = []  # 3d point in real world space
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((squares, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    for left_name, right_name in zip(left_images, right_images):
        left_img = cv2.imread(left_name)
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        left_ret, left_corners = cv2.findChessboardCorners(left_gray,
                                                           (rows, cols),
                                                           None)

        right_img = cv2.imread(right_name)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        right_ret, right_corners = cv2.findChessboardCorners(right_gray,
                                                             (rows, cols),
                                                             None)


        if left_ret and right_ret:
            objpoints.append(objp)
            left_corners2 = cv2.cornerSubPix(left_gray,
                                        left_corners,
                                        (11, 11),
                                        (-1, -1),
                                        intrinsic_criteria)
            imgpoints_left.append(left_corners2)

            right_corners2 = cv2.cornerSubPix(right_gray,
                                        right_corners,
                                        (11, 11),
                                        (-1, -1),
                                        intrinsic_criteria)
            imgpoints_right.append(right_corners2)

            if draw:
                left_img_cv = cv2.drawChessboardCorners(left_img,
                                                     (rows, cols),
                                                     left_corners2,
                                                     left_ret)
                cv2.imshow(os.path.split(left_name)[1], left_img_cv)

                right_img_cv = cv2.drawChessboardCorners(right_img,
                                                     (rows, cols),
                                                     right_corners2,
                                                     right_ret)
                cv2.imshow(os.path.split(right_name)[1], right_img_cv)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        return objpoints, imgpoints_left, imgpoints_right, \
               right_gray.shape[::-1]


def retrieve(filtered_images, draw, out_name):
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((squares, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    for image_name in filtered_images:
        img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), intrinsic_criteria)
            imgpoints.append(corners2)

            if draw:
                img = cv2.drawChessboardCorners(img, (rows, cols), corners2, ret)
                cv2.imshow(os.path.split(image_name)[1], img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                # plt.imshow(img)
                # plt.title(os.path.split(image_name)[1])
                # plt.pause(1)


    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None)

    camera_matrix = np.asarray(camera_matrix).tolist()
    dist_coefs = np.asarray(dist_coefs).tolist()

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs)

    # transform the matrix and distortion coefficients to writable lists
    data = {'camera_matrix': camera_matrix,
            'dist_coeff': dist_coefs,
            'RMS': rms}

    # and save it to a file
    folder_name = 'outputs'
    os.makedirs(folder_name, exist_ok=True)
    pn = os.path.join(folder_name, out_name + "_calibration_matrix.yaml")
    with open(pn, "w") as f:
        yaml.dump(data, f)
    return camera_matrix, dist_coefs,


image_dict = {}
for image_wc in ['left', 'right']:
    images = glob.glob(image_path + image_wc + '*.jpg')
    filtered_images = []
    for im in images:
        add = True
        for b in bad:
            if b in im:
                add = False
        if add:
            filtered_images.append(im)
    image_dict[image_wc] = filtered_images

    # # get camera matrix and distortion
    # matrix, dist_coefs = retrieve(filtered_images,
    #                               draw=False,
    #                               out_name=image_wc)
    #
    # # get undistorted camera matrix
    # img = cv2.imread(filtered_images[0])
    # img_height, img_width = img.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(np.array(matrix),
    #                                                   np.array(dist_coefs),
    #                                                   (img_width, img_height),
    #                                                   0,
    #                                                   (img_width, img_height))

    # x, y, w, h = roi
    #
    # for image_name in filtered_images:
    #     img = cv2.imread(image_name)
    #
    #     dst = cv2.undistort(img,
    #                         np.array(matrix),
    #                         np.array(dist_coefs),
    #                         None,
    #                         np.array(newcameramtx))
    #     cv2.imshow('original', img)
    #     cv2.imshow('undistorted', dst)
    #     cv2.waitKey(0)


objp, lpt, rpt, wh = stereo_retrieve(image_dict['left'],
                                     image_dict['right'],
                                     False)

lrms, lmat, ldist, lrotvec, ltransvec = cv2.calibrateCamera(
    objp,
    lpt,
    wh,
    None,
    None)

rrms, rmat, rdist, rrotvec, rtransvec = cv2.calibrateCamera(
    objp,
    rpt,
    wh,
    None,
    None)

stereo_retval, lmat1, ldist1, rmat1, rdist1, R, T, E, F = \
    cv2.stereoCalibrate(objp,
                        lpt,
                        rpt,
                        lmat,
                        ldist,
                        rmat,
                        rdist,
                        wh,
                        criteria=stereo_criteria,
                        flags=stereo_flags)

lproj = compute_projection_matrix(lmat1,
                                  rvec=lrotvec[0],
                                  tvec=ltransvec[0],
                                  rodrigues=True)

rproj = compute_projection_matrix(rmat1,
                                  rvec=rrotvec[0],
                                  tvec=rtransvec[0],
                                  rodrigues=True)

img = cv2.imread(image_dict['left'][0])

lpt_undist = cv2.undistortPoints(lpt[0], lmat1, ldist1, P=lmat1)
rpt_undist = cv2.undistortPoints(rpt[0], rmat1, rdist1, P=rmat1)

X = cv2.triangulatePoints(lproj,
                          rproj,
                          lpt_undist.squeeze().T,
                          rpt_undist.squeeze().T)
X /= X[3]
X = X[:3]
print(np.sum(np.square(np.diff(X)), axis=0))

plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[0], X[1], X[2], c=np.arange(X.shape[1]))
plt.xlim([0, 10])
plt.ylim([0, 10])
ax.set_zlim(-2, 2)
plt.show()


