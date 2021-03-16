#!/usr/bin/env python3

from tqdm import tqdm, trange
import os.path, os
import numpy as np
import pandas as pd
from numpy import array as arr
from glob import glob
from scipy import signal

from .common import make_process_fun, natural_keys

def medfilt_data(values, size=15):
    padsize = size+5
    vpad = np.pad(values, (padsize, padsize),
                  mode='median',
                  stat_length=5)
    vpadf = signal.medfilt(vpad, kernel_size=size)
    return vpadf[padsize:-padsize]

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_data(vals):
    nans, ix = nan_helper(vals)
    out = np.copy(vals)
    if np.mean(nans) > 0.8:
        return out
    out[nans] = np.interp(ix(nans), ix(~nans), vals[~nans])
    return out

def filter_pose(config, fname, outname):
    data = pd.read_csv(fname)
    cols = [x for x in data.columns if '_error' in x]
    bodyparts = [c.replace('_error', '') for c in cols]

    for bp in bodyparts:
        error = np.array(data[bp + '_error'])
        score = np.array(data[bp + '_score'])
        error[np.isnan(error)] = 100000
        score[np.isnan(score)] = -1
        good = score > config['score_threshold']
        # good = error < config['error_threshold']
        bad = np.invert(good)
        for v in 'xyz':
            key = '{}_{}'.format(bp, v)
            values = np.array(data[key])
            values[bad] = np.nan
            values_filt = interpolate_data(values)
            # values_filt = medfilt_data(values_filt, size=config['medfilt'])
            data[key] = values_filt
        data[bp+'_error'] = 1 # FIXME: hack for plotting
    data.to_csv(outname, index=False)

def process_peter(filter_config,
                  pose_3d_folder,
                  output_folder):
    os.makedirs(output_folder, exist_ok=True)
    pose_files = glob(os.path.join(pose_3d_folder, '*.csv'))
    pose_files = sorted(pose_files, key=natural_keys)

    for fname in pose_files:
        basename = os.path.basename(fname)
        outpath = os.path.join(output_folder, basename)
        filter_pose(filter_config, fname, outpath)

