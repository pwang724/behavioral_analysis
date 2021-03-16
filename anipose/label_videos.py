#!/usr/bin/env python3

import os.path
import numpy as np
from glob import glob
import pandas as pd
import cv2
import skvideo.io
from tqdm import trange
from .common import natural_keys

def connect(img, points, connectedparts, bodyparts, color):
    ixs = [bodyparts.index(bp) for bp in connectedparts]
    for a, b in zip(ixs, ixs[1:]):
        if np.any(np.isnan(points[[a,b]])):
            continue
        pa = tuple(np.int32(points[a]))
        pb = tuple(np.int32(points[b]))
        cv2.line(img,
                 pt1=tuple(pa),
                 pt2=tuple(pb),
                 color=color,
                 thickness=1)

def connect_all(img, points, scheme, bodyparts):
    for connectedparts in scheme:
        col = [255, 255, 255, 255]
        connect(img, points, connectedparts, bodyparts, col)


def label_frame(img, points, scheme, bodyparts, color_dict, size_dict):
    connect_all(img, points, scheme, bodyparts)
    for lnum, (x, y) in enumerate(points):
        if np.isnan(x) or np.isnan(y):
            continue
        x = np.clip(x, 1, img.shape[1]-1)
        y = np.clip(y, 1, img.shape[0]-1)
        x = int(round(x))
        y = int(round(y))
        col = color_dict[bodyparts[lnum]][:3]
        size = size_dict[bodyparts[lnum]]
        cv2.circle(img=img,
                   center=(x,y),
                   radius=size,
                   color=col.astype(float),
                   thickness=size)
    return img

def visualize_labels(scheme,
                     threshold,
                     color_dict,
                     size_dict,
                     labels_fname,
                     vid_fname,
                     out_fname):
    dlabs = pd.read_hdf(labels_fname)
    if len(dlabs.columns.levels) > 2:
        scorer = dlabs.columns.levels[0][0]
        dlabs = dlabs.loc[:, scorer]

    if scheme:
        bodyparts = sorted(set([x for dx in scheme for x in dx]))
    else:
        bodyparts = list(dlabs.columns.levels[0])

    cap = cv2.VideoCapture(vid_fname)
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = skvideo.io.FFmpegWriter(out_fname,
                                     inputdict={
                                         '-hwaccel': 'auto',
                                         '-framerate': str(fps),
                                     },
                                     outputdict={
                                         '-vcodec': 'mjpeg',
                                         '-qp': '28',
                                         '-qscale': '0',
                                         '-pix_fmt': 'yuvj420p',
                                     })

    points = [(dlabs[bp]['x'], dlabs[bp]['y']) for bp in bodyparts]
    points = np.array(points)
    scores = [dlabs[bp]['likelihood'] for bp in bodyparts]
    scores = np.array(scores)
    scores[np.isnan(scores)] = 0
    scores[np.isnan(points[:, 0])] = 0
    good = np.array(scores) > threshold
    points[:, 0, :][~good] = np.nan
    points[:, 1, :][~good] = np.nan

    all_points = points
    for ix in trange(len(dlabs), ncols=70):
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        points = all_points[:, :, ix]
        img = label_frame(img, points, scheme, bodyparts, color_dict, size_dict)
        writer.writeFrame(img)
    cap.release()
    writer.close()


def process_peter(scheme,
                  threshold,
                  body_part_colors,
                  body_part_sizes,
                  video_folder,
                  pose_2d_folder,
                  out_folder,
                  video_type='avi'):
    os.makedirs(out_folder, exist_ok=True)
    label_pns = glob(os.path.join(pose_2d_folder, '*.h5'))
    label_pns = sorted(label_pns, key=natural_keys)
    for label_pn in label_pns:
        base_name = os.path.basename(label_pn)
        base_name = os.path.splitext(base_name)[0]
        video_name = os.path.join(video_folder, base_name + '.' + video_type)
        out_fname = os.path.join(out_folder, base_name + '.avi')
        if not os.path.exists(video_name):
            raise ValueError(f'{video_name} does not exist, but {label_pn} exists')
        visualize_labels(scheme=scheme,
                         threshold=threshold,
                         color_dict=body_part_colors,
                         size_dict=body_part_sizes,
                         labels_fname=label_pn,
                         vid_fname=video_name,
                         out_fname=out_fname)

