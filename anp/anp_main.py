import matplotlib.colors

import anipose as anp
import anipose.anipose
import anipose.common
import anipose.label_videos_3d
import anipose.label_combined
import os
import numpy as np
import tools
import anipose.pose_videos
import anipose.label_videos
import anipose.filter_pose
import anipose.filter_3d
import anipose.calibrate
import anipose.triangulate
import base
import video_tools
from base import SCORE_THRESHOLD

FILTER_CONFIG = {
    'enabled': False,
    'type': 'medfilt',
    'medfilt': 13,
    'offset_threshold': 25,
    'score_threshold': SCORE_THRESHOLD,
    'spline': True,
    'n_back': 5,
    'multiprocessing': False
}
FILTER3D_CONFIG = {
    'error_threshold': 50,
    'score_threshold': 0.95,
    'medfilt': 13
}

triangulation_config = {
    'ransac': False,
    'optim': True,
    'scale_smooth': 1,
    'scale_length': 1,
    'scale_length_weak': 0.5,
    'reproj_error_threshold': 5,
    'score_threshold': 0.5,
    'n_deriv_smooth': 2,
    'constraints': [
        ["r1_in", "r1_out"],
        ["r2_in", "r2_out"],
        ["r3_in", "r3_out"],
        ["r4_in", "r4_out"],
        ["r1_in", "r2_in"],
        ["r2_in", "r3_in"],
        ["r3_in", "r4_in"]],
    'constraints_weak': [],
    'use_saved_reference': []
}

scheme = [['r1_in', 'r1_out'],
          ['r2_in', 'r2_out'],
          ['r3_in', 'r3_out'],
          ['r4_in', 'r4_out'],
          ['r1_in', 'r2_in', 'r3_in', 'r4_in'],
          ['pellet'],
          ['insured pellet']]

colors = {
    'r1_in': 'red',
    'r1_out': 'lightsalmon',
    'r2_in': 'orange',
    'r2_out': 'bisque',
    'r3_in': 'green',
    'r3_out': 'lightgreen',
    'r4_in': 'blue',
    'r4_out': 'lightblue',
    'pellet': 'aqua',
    'insured pellet': 'brown'
}

for k, v in colors.items():
    col = matplotlib.colors.to_rgba_array(v)[0] * 255
    colors[k] = np.array([int(c) for c in col])

sizes = {
    'r1_in': 1,
    'r1_out': 1,
    'r2_in': 1,
    'r2_out': 1,
    'r3_in': 1,
    'r3_out': 1,
    'r4_in': 1,
    'r4_out': 1,
    'pellet': 3,
    'insured pellet': 3
}

body_parts = np.unique([x for y in scheme for x in y])


if __name__ == '__main__':
    # basename = r'C:\Users\Peter\Desktop\anipose\checkerboard-unfilled'
    # fname = os.path.join(basename, r'config.toml')
    # cfg = anp.anipose.load_config(basename)
    # anp.calibrate.process_peter(calibration_path=calib_dir,
    #                             board=base.CALIBRATION_BOARD,
    #                             cam_regex=base.CAM_REGEX,
    #                             fisheye=False,
    #                             video_ext=base.RAW_VIDEO_EXT)

    def analyze(input_folder, model_dirs, test):
        path, date = os.path.split(input_folder)
        path, mouse = os.path.split(path)
        desktop = tools.desktop_path()
        analyzed_folder = os.path.join(desktop, 'ANALYZED', mouse, date)

        pose2d_folder = os.path.join(analyzed_folder, 'POSE_2D')
        pose2d_filter_folder = os.path.join(analyzed_folder, 'POSE_2D_FILTERED')
        pose2d_video_folder = os.path.join(analyzed_folder, 'VIDEOS_2D')
        dlc_collage_folder = os.path.join(analyzed_folder, 'COLLAGE_DLC')
        pose2d_video_filtered_folder = os.path.join(analyzed_folder,
                                                    'VIDEOS_2D_FILTERED')
        pose3d_folder = os.path.join(analyzed_folder, 'POSE_3D')
        pose3d_filter_folder = os.path.join(analyzed_folder, 'POSE_3D_FILTER')
        pose3d_video_folder = os.path.join(analyzed_folder, 'VIDEOS_3D')
        combined_video_folder = os.path.join(analyzed_folder,
                                             'VIDEOS_3D_COMBINED')
        dict_of_avis = {x: tools.get_files(input_folder, (x + '.' + base.RAW_VIDEO_EXT))
                        for x in base.CAMERA_NAMES}
        #TEMPORARY
        import shutil
        if os.path.exists(pose2d_video_folder):
            shutil.rmtree(pose2d_video_folder)
        if os.path.exists(pose2d_folder):
            shutil.rmtree(pose2d_folder)
        if os.path.exists(dlc_collage_folder):
            shutil.rmtree(dlc_collage_folder)

        if test:
            for k, v in dict_of_avis.items():
                dict_of_avis[k] = v[:1]

        for cam, video_pns in dict_of_avis.items():
            model = model_dirs[cam]
            anp.pose_videos.process_peter(videos=video_pns,
                                          model_folder=model,
                                          out_folder=pose2d_folder,
                                          video_type=base.RAW_VIDEO_EXT)

        anp.label_videos.process_peter(scheme=scheme,
                                       threshold=base.SCORE_THRESHOLD,
                                       body_part_colors=colors,
                                       body_part_sizes=sizes,
                                       video_folder=input_folder,
                                       pose_2d_folder=pose2d_folder,
                                       out_folder= pose2d_video_folder,
                                       video_type='avi'
                                       )

        videos0 = [os.path.join(pose2d_video_folder, x) for x in os.listdir(
            pose2d_video_folder)
                   if 'CAM0' in x]
        videos0 = sorted(videos0)
        video_tools.make_collage(videos0,
                                 width=5,
                                 height=4,
                                 collage_folder=dlc_collage_folder,
                                 savestr='CAM0')

        # anp.filter_pose.process_peter(FILTER_CONFIG,
        #                               pose2d_folder,
        #                               pose2d_filter_folder)
        #
        # anp.label_videos.process_peter(scheme=scheme,
        #                                threshold=base.SCORE_THRESHOLD,
        #                                body_part_colors=colors,
        #                                body_part_sizes=sizes,
        #                                video_folder=input_folder,
        #                                pose_2d_folder=pose2d_filter_folder,
        #                                out_folder= pose2d_video_filtered_folder
        #                                )
        #
        # anp.triangulate.process_peter(triangulation_config=triangulation_config,
        #                               calib_folder=calib_dir,
        #                               pose_folder=pose2d_folder,
        #                               video_folder=input_folder,
        #                               output_folder=pose3d_folder,
        #                               cam_regex=base.CAM_REGEX)
        # anp.filter_3d.process_peter(FILTER3D_CONFIG,
        #                             pose3d_folder,
        #                             pose3d_filter_folder)
        # anp.label_videos_3d.process_peter(scheme=scheme,
        #                                   optim=triangulation_config['optim'],
        #                                   video_folder=input_folder,
        #                                   pose_3d_folder=pose3d_folder,
        #                                   out_folder=pose3d_video_folder,
        #                                   video_ext=base.RAW_VIDEO_EXT,
        #                                   cam_regex=base.CAM_REGEX)
        # anp.label_combined.process_peter(scheme=scheme,
        #                                  optim=triangulation_config['optim'],
        #                                  calib_folder=calib_dir,
        #                                  video_folder=input_folder,
        #                                  pose_3d_folder=pose3d_folder,
        #                                  video_3d_folder=pose3d_video_folder,
        #                                  out_folder=combined_video_folder,
        #                                  cam_regex=base.CAM_REGEX,
        #                                  video_ext=base.RAW_VIDEO_EXT)

    calib_dir = r'C:\Users\Peter\Desktop\anipose\calibration'
    data_dir = r'C:\Users\Peter\Desktop\DATA'
    model_dirs = {
        'CAM0': r'C:\Users\Peter\Desktop\DLC\M2_M4_M5_M9_FRONT-pw-2021-03-08',
        'CAM1': r'C:\Users\Peter\Desktop\DLC\M2_M4_M5_M9_SIDE-pw-2021-03-09'
    }
    mouse_dates = [
        r'M5\2021_03_11',
        r'M5\2021_03_12',
        r'M5\2021_03_14',
        # r'M9\2021.03.11',
        # r'M9\2021.03.12',
        # r'M9\2021.03.14',
    ]
    test = False
    for mouse_date in mouse_dates:
        input_folder = os.path.join(data_dir, mouse_date)
        analyze(input_folder, model_dirs, test=test)
