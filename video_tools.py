import glob
import os

import imageio
import numpy as np
from moviepy.editor import VideoFileClip, clips_array
from moviepy.video.fx.all import crop
import tools
import base
import skvideo.io

def make_collage(videos0, width, height, collage_folder, savestr,
                 cropvid=True,
                 filext='mp4'):
    os.makedirs(collage_folder, exist_ok=True)
    n_collages = int(np.ceil(len(videos0)/(width*height)))
    for n in range(n_collages):
        start_ix = n * width * height
        pn = os.path.join(collage_folder,
                          f'{savestr}_'
                          f'{start_ix}-{start_ix + width*height-1}.{filext}')
        if os.path.exists(pn):
            print(f'EXISTS: {pn}')
            continue

        clips_arr = []
        for h in range(height):
            temp = []
            for w in range(width):
                ix = h * width + w + start_ix
                if ix < len(videos0):
                    clip = VideoFileClip(videos0[ix])
                    if cropvid:
                        clip = crop(clip, x1=100, y1=0, x2=350, y2=190)
                    clip = clip.resize(width=125)
                    temp.append(clip)
                else:
                    clip = VideoFileClip(videos0[-1])
                    if cropvid:
                        clip = crop(clip, x1=100, y1=0, x2=350, y2=190)
                    clip = clip.resize(width=5)
                    temp.append(clip)

            clips_arr.append(temp)
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


def mp4_to_avi(d, delete_old=True):
    mp4_files = sorted(glob.glob(d + '/*.mp4'))

    for mp4 in mp4_files:
        reader = imageio.get_reader(mp4)
        fps = reader.get_meta_data()['fps']
        writer = imageio.get_writer(os.path.splitext(mp4)[0] + '.avi',
                                    fps=fps,
                                    codec='mjpeg',
                                    pixelformat='yuvj420p',
                                   quality=10)

        for im in reader:
            writer.append_data(im[:, :, :])
        writer.close()
    if delete_old:
        for mp4 in mp4_files:
            os.remove(mp4)

def avi_to_mp4(d, delete_old=True):
    avi_files = sorted(glob.glob(d + '/*.avi'))

    for avi in avi_files:
        reader = imageio.get_reader(avi)
        fps = reader.get_meta_data()['fps']
        writer = skvideo.io.FFmpegWriter(os.path.splitext(avi)[0] + '.mp4',
                                         inputdict={
                                             '-hwaccel': 'auto',
                                             '-framerate': str(fps),
                                         },
                                         outputdict={
                                             '-vcodec': 'h264',
                                             '-qp': '25',
                                             '-qscale': '0',
                                             '-pix_fmt': 'yuv420p',
                                         })

        for im in reader:
            writer.writeFrame(im[:, :, :])
        writer.close()
    if delete_old:
        for avi in avi_files:
            os.remove(avi)