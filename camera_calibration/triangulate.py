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
for image_wc in ['left', 'right']:
    images = [im for im in os.listdir(c.IMAGE_DIR) if c.IM_EXTENSION in im and
              image_wc in im]
    bad_images = [image_wc + ix + '.jpg' for ix in bad_ixs]
    good_images = sorted(list(set(images) - set(bad_images)))
    image_dict[image_wc] = [os.path.join(c.IMAGE_DIR, x) for x in good_images]

objp, lpt, rpt, wh = tools.get_stereo_points(image_dict['left'],
                                             image_dict['right'],
                                             draw=False)

# Triangulate
lpt_undist = cv2.undistortPoints(lpt[0], lmat, ldist, P=lmat)
rpt_undist = cv2.undistortPoints(rpt[0], rmat, rdist, P=rmat)

X = cv2.triangulatePoints(lproj,
                          rproj,
                          lpt_undist.squeeze().T,
                          rpt_undist.squeeze().T)
X /= X[3]
X = X[:3]
print(np.sum(np.square(np.diff(X)), axis=0))

plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[0], X[1], -X[2], c=np.arange(X.shape[1]), cmap='RdBu')
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
plt.xlim([-10, 10])
plt.ylim([-10, 10])
ax.set_zlim(-10, 10)
plt.show()
