import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os
import camera_calibration.constants as c
import camera_calibration.tools as tools

# LOAD CAMERA SETTINGS
pkl_path = r'..\camera_calibration/outputs/intrinsic_calibration.pkl'
with open(pkl_path, 'rb') as file:
    calib_dict = pickle.load(file)

lmat = calib_dict['lmat']
ldist = calib_dict['ldist']
lproj = calib_dict['lproj']
rmat = calib_dict['rmat']
rdist = calib_dict['rdist']
rproj = calib_dict['rproj']

# Load data
bad_ixs = [str(x).zfill(2) for x in np.arange(11, 20)]
image_dict = {}
for image_wc in [c.CAM_ONE_ORIENT, c.CAM_TWO_ORIENT]:
    images = [im for im in os.listdir(c.IM_DIR) if c.IM_EXTENSION in im and
              image_wc in im]
    bad_images = [image_wc + ix + '.jpg' for ix in bad_ixs]
    good_images = sorted(list(set(images) - set(bad_images)))
    image_dict[image_wc] = [os.path.join(c.IM_DIR, x) for x in good_images]

objp, lpt, rpt, wh = tools.get_stereo_points(image_dict[c.CAM_ONE_ORIENT],
                                             image_dict[c.CAM_TWO_ORIENT],
                                             draw=False)

# Triangulate
plt.figure()
ax = plt.axes(projection='3d')

for i in range(10):
    lpt_undist = cv2.undistortPoints(lpt[i], lmat, ldist, P=lmat)
    rpt_undist = cv2.undistortPoints(rpt[i], rmat, rdist, P=rmat)

    X = cv2.triangulatePoints(lproj,
                              rproj,
                              lpt_undist.squeeze().T,
                              rpt_undist.squeeze().T)
    X /= X[3]
    X = X[:3]

    print(np.sum(np.square(np.diff(X)), axis=0))

    mat = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float32)
    X = np.matmul(mat, X)

    # ax.scatter3D(X[0], X[1], X[2], c=np.arange(X.shape[1]), cmap='RdBu')
    ax.plot3D(X[0], X[1], X[2])

plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
plt.xlim([-15, 15])
plt.ylim([-15, 15])
ax.set_zlim(-15, 15)
plt.show()
