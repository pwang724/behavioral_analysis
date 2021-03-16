import anipose as anp
import anipose.label_videos_3d
import anipose.label_combined
import os
import numpy as np
import tools

from aniposelib.boards import Checkerboard

DATA_DIR = r'C:\Users\Peter\Desktop\DLC\DATA'
CALIB_DIR = r'C:\Users\Peter\Desktop\anipose\calibration'
MOUSE_DIR =  'M2'
DATE_DIR = '2021_03_14'
CAMERA_NAMES = ['CAM0', 'CAM1']
CAM_REGEX = 'CAM([0-9])'
MODEL_DIRS = {
    'CAM0': r'C:\Users\Peter\Desktop\DLC\M2_M4_M5_M9_FRONT-pw-2021-03-08',
    'CAM1': r'C:\Users\Peter\Desktop\DLC\M2_M4_M5_M9_SIDE-pw-2021-03-09'
}
VIDEO_EXT = 'avi'
DLC_LIKELIHOOD_THRES = 0.999

filter_config = {
    'enabled': False,
    'type': 'medfilt',
    'medfilt': 13,
    'offset_threshold': 25,
    'score_threshold': DLC_LIKELIHOOD_THRES,
    'spline': True,
    'n_back': 5,
    'multiprocessing': False
}

filter3d_config = {
    'error_threshold': 50,
    'score_threshold': 0.9,
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


board = Checkerboard(squaresX=9,
                     squaresY=6,
                     square_length=1,
                     manually_verify=False)


body_parts = np.unique([x for y in scheme for x in y])

input_folder = os.path.join(DATA_DIR, MOUSE_DIR, DATE_DIR)
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
    # fname = r'C:\Users\Peter\Desktop\anipose\reach-unfilled\config.toml'
    # fname = r'C:\Users\Peter\Desktop\anipose\checkerboard-unfilled\config.toml'
    # cfg = load_config(fname)
    # img = anipose.common.get_calibration_board_image(cfg)
    # cv2.imwrite('calibration.png', img)


    # for cam, video_pns in dict_of_avis.items():
    #     model = MODEL_DIRS[cam]
    #     anp.pose_videos.process_peter(videos=video_pns,
    #                                   model_folder=model,
    #                                   out_folder=pose2d_folder,
    #                                   video_type=VIDEO_EXT)
    #
    # anp.label_videos.process_peter(scheme=scheme,
    #                                threshold=DLC_LIKELIHOOD_THRES,
    #                                body_part_colors=constants.colors,
    #                                body_part_sizes=constants.sizes,
    #                                video_folder=input_folder,
    #                                pose_2d_folder=pose2d_folder,
    #                                out_folder= pose2d_video_folder
    #                                )
    # anp.filter_pose.process_peter(filter_config,
    #                               pose2d_folder,
    #                               pose2d_filter_folder)
    #
    # anp.label_videos.process_peter(scheme=scheme,
    #                                threshold=DLC_LIKELIHOOD_THRES,
    #                                body_part_colors=constants.colors,
    #                                body_part_sizes=constants.sizes,
    #                                video_folder=input_folder,
    #                                pose_2d_folder=pose2d_filter_folder,
    #                                out_folder= pose2d_video_filtered_folder
    #                                )
    # anp.calibrate.process_peter(calibration_path=CALIB_DIR,
    #                             board=board,
    #                             cam_regex=CAM_REGEX,
    #                             fisheye=False,
    #                             video_ext=VIDEO_EXT)
    # anp.triangulate.process_peter(triangulation_config=triangulation_config,
    #                               calib_folder=CALIB_DIR,
    #                               pose_folder=pose2d_folder,
    #                               video_folder=input_folder,
    #                               output_folder=pose3d_folder,
    #                               cam_regex=CAM_REGEX)
    # anp.filter_3d.process_peter(filter3d_config,
    #                             pose3d_folder,
    #                             pose3d_filter_folder)
    anp.label_videos_3d.process_peter(scheme=scheme,
                                      optim=triangulation_config['optim'],
                                      video_folder=input_folder,
                                      pose_3d_folder=pose3d_folder,
                                      out_folder=pose3d_video_folder,
                                      video_ext=VIDEO_EXT,
                                      cam_regex=CAM_REGEX)
    anp.label_combined.process_peter(scheme=scheme,
                                     optim=triangulation_config['optim'],
                                     calib_folder=CALIB_DIR,
                                     video_folder=input_folder,
                                     pose_3d_folder=pose3d_folder,
                                     video_3d_folder=pose3d_video_folder,
                                     out_folder=combined_video_folder,
                                     cam_regex=CAM_REGEX,
                                     video_ext=VIDEO_EXT)