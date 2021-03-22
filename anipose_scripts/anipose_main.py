import matplotlib.colors

import anipose as anp
import anipose.label_videos_3d
import anipose.label_combined
import os
import numpy as np
import tools
import anipose_scripts.constants as constants
import anipose.pose_videos
import anipose.label_videos
import anipose.filter_pose
import anipose.filter_3d
import anipose.calibrate
import anipose.triangulate

from anipose_scripts.constants import VIDEO_EXT, DLC_LIKELIHOOD_THRES, \
    FILTER_CONFIG, FILTER3D_CONFIG, CAMERA_NAMES, CAM_REGEX, CALIBRATION_BOARD

calib_dir = r'C:\Users\Peter\Desktop\anipose\calibration'
data_dir = r'C:\Users\Peter\Desktop\anipose\TEST_DATA'
mouse_dir = 'TEST_MOUSE'
date_dir = 'TEST_DATE'
model_dirs = {
    'CAM0': r'C:\Users\Peter\Desktop\DLC\M2_M4_M5_M9_FRONT-pw-2021-03-08',
    'CAM1': r'C:\Users\Peter\Desktop\DLC\M2_M4_M5_M9_SIDE-pw-2021-03-09'
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

input_folder = os.path.join(data_dir, mouse_dir, date_dir)
pose2d_folder = os.path.join(input_folder,
                             'ANALYZED',
                             'POSE_2D')
pose2d_filter_folder = os.path.join(input_folder,
                                    'ANALYZED',
                                    'POSE_2D_FILTERED')
pose2d_video_folder = os.path.join(input_folder,
                                   'ANALYZED',
                                   'VIDEOS_2D')
pose2d_video_filtered_folder = os.path.join(input_folder,
                                            'ANALYZED',
                                            'VIDEOS_2D_FILTERED')
pose3d_folder = os.path.join(input_folder,
                             'ANALYZED',
                             'POSE_3D')
pose3d_filter_folder = os.path.join(input_folder,
                                    'ANALYZED',
                                    'POSE_3D_FILTER')
pose3d_video_folder = os.path.join(input_folder,
                                            'ANALYZED',
                                            'VIDEOS_3D')
combined_video_folder = os.path.join(input_folder,
                                            'ANALYZED',
                                            'VIDEOS_3D_COMBINED')
dict_of_avis = {x: tools.get_files(input_folder, (x + '.' + VIDEO_EXT))
                for x in CAMERA_NAMES}

if __name__ == '__main__':
    fname = r'C:\Users\Peter\Desktop\anipose\reach-unfilled\config.toml'
    fname = r'C:\Users\Peter\Desktop\anipose\checkerboard-unfilled\config.toml'
    # cfg = load_config(basename)
    # img = anipose.common.get_calibration_board_image(cfg)
    # cv2.imwrite('calibration.png', img)

    # anp.calibrate.process_peter(calibration_path=calib_dir,
    #                             board=CALIBRATION_BOARD,
    #                             cam_regex=CAM_REGEX,
    #                             fisheye=False,
    #                             video_ext=VIDEO_EXT)
    #
    # for cam, video_pns in dict_of_avis.items():
    #     model = model_dirs[cam]
    #     anp.pose_videos.process_peter(videos=video_pns,
    #                                   model_folder=model,
    #                                   out_folder=pose2d_folder,
    #                                   video_type=VIDEO_EXT)
    # #
    # anp.label_videos.process_peter(scheme=scheme,
    #                                threshold=DLC_LIKELIHOOD_THRES,
    #                                body_part_colors=colors,
    #                                body_part_sizes=sizes,
    #                                video_folder=input_folder,
    #                                pose_2d_folder=pose2d_folder,
    #                                out_folder= pose2d_video_folder
    #                                )
    #
    # anp.filter_pose.process_peter(FILTER_CONFIG,
    #                               pose2d_folder,
    #                               pose2d_filter_folder)

    anp.label_videos.process_peter(scheme=scheme,
                                   threshold=DLC_LIKELIHOOD_THRES,
                                   body_part_colors=colors,
                                   body_part_sizes=sizes,
                                   video_folder=input_folder,
                                   pose_2d_folder=pose2d_filter_folder,
                                   out_folder= pose2d_video_filtered_folder
                                   )

    # anp.triangulate.process_peter(triangulation_config=triangulation_config,
    #                               calib_folder=calib_dir,
    #                               pose_folder=pose2d_folder,
    #                               video_folder=input_folder,
    #                               output_folder=pose3d_folder,
    #                               cam_regex=CAM_REGEX)
    # anp.filter_3d.process_peter(FILTER3D_CONFIG,
    #                             pose3d_folder,
    #                             pose3d_filter_folder)
    # anp.label_videos_3d.process_peter(scheme=scheme,
    #                                   optim=triangulation_config['optim'],
    #                                   video_folder=input_folder,
    #                                   pose_3d_folder=pose3d_folder,
    #                                   out_folder=pose3d_video_folder,
    #                                   video_ext=VIDEO_EXT,
    #                                   cam_regex=CAM_REGEX)
    # anp.label_combined.process_peter(scheme=scheme,
    #                                  optim=triangulation_config['optim'],
    #                                  calib_folder=calib_dir,
    #                                  video_folder=input_folder,
    #                                  pose_3d_folder=pose3d_folder,
    #                                  video_3d_folder=pose3d_video_folder,
    #                                  out_folder=combined_video_folder,
    #                                  cam_regex=CAM_REGEX,
    #                                  video_ext=VIDEO_EXT)