import os
import glob
import imageio
import numpy as np
import pandas as pd
import re

d = r'C:\Users\Peter\Desktop\DLC\new_position-pw-2021-03-01\videos_pred_dlc'

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


def get_files(source, wcs):
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith(wcs):
                matches.append(os.path.join(root, filename))
    return matches


def print_dict(ddict, length=True):
    for k, v in ddict.items():
        if length:
            print(f'{k}: {len(v)}')
        else:
            print(f'{k}: {v}')


def load_pose_2d(fname):
    data_orig = pd.read_hdf(fname)
    scorer = data_orig.columns.levels[0][0]
    data = data_orig.loc[:, scorer]

    bp_index = data.columns.names.index('bodyparts')
    coord_index = data.columns.names.index('coords')
    bodyparts = list(data.columns.get_level_values(bp_index).unique())
    n_possible = len(data.columns.levels[coord_index])//3

    n_frames = len(data)
    n_joints = len(bodyparts)
    test = np.array(data).reshape(n_frames, n_joints, n_possible, 3)

    metadata = {
        'bodyparts': bodyparts,
        'scorer': scorer,
        'index': data.index
    }
    return test, metadata


def write_pose_2d(all_points, metadata, outname=None):
    points = all_points[:, :, :2]
    scores = all_points[:, :, 2]

    scorer = metadata['scorer']
    bodyparts = metadata['bodyparts']
    index = metadata['index']

    columns = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'])

    dout = pd.DataFrame(columns=columns, index=index)

    dout.loc[:, (scorer, bodyparts, 'x')] = points[:, :, 0]
    dout.loc[:, (scorer, bodyparts, 'y')] = points[:, :, 1]
    dout.loc[:, (scorer, bodyparts, 'likelihood')] = scores

    if outname is not None:
        dout.to_hdf(outname, 'df_with_missing', format='table', mode='w')

    return dout


def true_basename(fname):
    basename = os.path.basename(fname)
    basename = os.path.splitext(basename)[0]
    return basename

def get_cam_name(cam_regex, fname):
    basename = true_basename(fname)
    match = re.search(cam_regex, basename)
    if not match:
        return None
    else:
        name = match.groups()[0]
        return name.strip()

def get_video_name(cam_regex, fname):
    basename = true_basename(fname)
    vidname = re.sub(cam_regex, '', basename)
    return vidname.strip()