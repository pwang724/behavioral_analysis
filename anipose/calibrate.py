#!/usr/bin/env python3

import os
from glob import glob
from collections import defaultdict
import pickle
import tools
from aniposelib.cameras import CameraGroup


def process_peter(calibration_path,
                  board,
                  cam_regex,
                  fisheye=False,
                  video_ext='avi'):
    videos = glob(os.path.join(calibration_path,'*.' +video_ext))
    videos = sorted(videos)

    cam_videos = defaultdict(list)
    cam_names = set()
    for vid in videos:
        name = tools.camname_from_regex(cam_regex, vid)
        cam_videos[name].append(vid)
        cam_names.add(name)
    cam_names = sorted(cam_names)
    cgroup = CameraGroup.from_names(cam_names, fisheye)

    video_list = [sorted(cam_videos[cname]) for cname in cam_names]
    if len(videos) == 0:
        print('no videos or calibration file found, continuing...')
        return

    rows_fname = os.path.join(calibration_path, 'detections.pickle')
    if os.path.exists(rows_fname):
        with open(rows_fname, 'rb') as f:
            all_rows = pickle.load(f)
    else:
        all_rows = cgroup.get_rows_videos(video_list, board)
        with open(rows_fname, 'wb') as f:
            pickle.dump(all_rows, f)

    cgroup.set_camera_sizes_videos(video_list)

    cgroup.calibrate_rows(all_rows, board,
                          init_extrinsics=True,
                          init_intrinsics=True,
                          max_nfev=100, n_iters=2,
                          n_samp_iter=100, n_samp_full=300,
                          verbose=True)
    error = cgroup.calibrate_rows(all_rows, board,
                                  init_intrinsics=False, init_extrinsics=False,
                                  max_nfev=100, n_iters=10,
                                  n_samp_iter=100, n_samp_full=1000,
                                  verbose=True)

    cgroup.metadata['adjusted'] = False
    if error is not None:
        cgroup.metadata['error'] = float(error)
    outname = os.path.join(calibration_path, 'calibration.toml')
    cgroup.dump(outname)
    print(outname)


