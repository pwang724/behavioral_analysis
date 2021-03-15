import anipose as anp
from anipose.anipose import load_config
import anipose.pose_videos
import anipose.filter_pose
import anipose.label_videos
import anipose.calibrate
import anipose.triangulate
import anipose.common
import anipose.label_videos_3d
import anipose.label_combined
import cv2
import pprint

fname = r'C:\Users\Peter\Desktop\anipose\reach-unfilled\config.toml'
# fname = r'C:\Users\Peter\Desktop\anipose\checkerboard-unfilled\config.toml'
cfg = load_config(fname)
# img = anipose.common.get_calibration_board_image(cfg)
# cv2.imwrite('calibration.png', img)
#
# anp.pose_videos.pose_videos_all(config=cfg)
anp.label_videos.label_videos_all(config=cfg)
#
# anp.filter_pose.filter_pose_all(config=cfg)
# anp.label_videos.label_videos_filtered_all(config=cfg)
#
# anp.calibrate.calibrate_all(config=cfg)

anp.triangulate.triangulate_all(config=cfg)
anp.label_videos_3d.label_videos_3d_all(config=cfg)
anp.label_combined.label_combined_all(config=cfg)