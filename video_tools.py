import os
import numpy as np
from moviepy.editor import VideoFileClip, clips_array
from moviepy.video.fx.all import crop
import tools
import base

def make_collage(videos0, width, height, collage_folder):
    os.makedirs(collage_folder, exist_ok=True)
    n_collages = int(np.ceil(len(videos0)/(width*height)))
    for n in range(n_collages):
        start_ix = n * width * height
        clips_arr = []
        for h in range(height):
            temp = []
            for w in range(width):
                ix = h * width + w + start_ix
                if ix < len(videos0):
                    clip = VideoFileClip(videos0[ix])
                    clip = crop(clip, x1=100, y1=0, x2=350, y2=190)
                    clip = clip.resize(width=125)
                    temp.append(clip)
                else:
                    clip = VideoFileClip(videos0[-1])
                    clip = crop(clip, x1=100, y1=0, x2=350, y2=190)
                    clip = clip.resize(width=5)
                    temp.append(clip)

            clips_arr.append(temp)
        pn = os.path.join(collage_folder,
                          f'{start_ix}-{start_ix + width*height-1}.mp4')
        final_clip = clips_array(clips_arr)
        final_clip.write_videofile(pn, codec='libx264')

def merge_cams(videos0, videos1, merged_folder):
    os.makedirs(merged_folder, exist_ok=True)
    for v0, v1 in zip(videos0, videos1):
        vidname = tools.videoname_from_regex(base.CAM_REGEX, v0)
        pn = os.path.join(merged_folder, vidname + '.mp4')
        if not os.path.exists(pn):
            clip1 = VideoFileClip(v0)
            clip2 = VideoFileClip(v1)
            final_clip = clips_array([[clip1, clip2]])
            final_clip.write_videofile(pn, codec='libx264')