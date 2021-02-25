import numpy as np
import cv2
import os
import pickle
import yaml
import camera_calibration.constants as c
import camera_calibration.tools as tools


bad_ixs = []

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

lrms, lmat, ldist, lrot, ltrans = cv2.calibrateCamera(
    objp,
    lpt,
    wh,
    None,
    None)

rrms, rmat, rdist, rrot, rtrans = cv2.calibrateCamera(
    objp,
    rpt,
    wh,
    None,
    None)

# Stereo calibration
stereo_retval, lmat1, ldist1, rmat1, rdist1, R, T, E, F = \
    cv2.stereoCalibrate(objp,
                        lpt,
                        rpt,
                        lmat,
                        ldist,
                        rmat,
                        rdist,
                        wh,
                        criteria=c.STEREO_CRIT,
                        flags=c.STEREO_FLAGS)

# Compute projection
reference_ix = 0
lproj = tools.compute_projection_matrix(lmat,
                                        rvec=lrot[0],
                                        tvec=ltrans[0],
                                        rodrigues=True)

rproj = tools.compute_projection_matrix(rmat,
                                        rvec=rrot[0],
                                        tvec=rtrans[0],
                                        rodrigues=True)

yaml_dict = {'lrms': lrms,
            'lmat': lmat1.tolist(),
            'ldist': ldist1.tolist(),
            'rrms': rrms,
            'rmat': rmat.tolist(),
            'rdist': rdist.tolist(),
            'stereo_rms': stereo_retval
             }

pkl_dict = {'lmat': lmat,
            'ldist': ldist,
            'lrot': lrot,
            'ltrans': ltrans,
            'lproj': lproj,
            'rmat': rmat1,
            'rdist': rdist1,
            'rrot': rrot,
            'rtrans': rtrans,
            'rproj': rproj
             }

os.makedirs(c.OUTPUT_FOLDER_NAME, exist_ok=True)
pn = os.path.join(c.OUTPUT_FOLDER_NAME, "intrinsic_calibration.yaml")
with open(pn, "w") as f:
    yaml.dump(yaml_dict, f)

pn = os.path.join(c.OUTPUT_FOLDER_NAME, "intrinsic_calibration.pkl")
with open(pn, "wb") as f:
    pickle.dump(pkl_dict, f)
