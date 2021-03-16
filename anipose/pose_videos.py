#!/usr/bin/env python3

import os.path
import os
from glob import glob
import io
from contextlib import redirect_stdout

def rename_dlc_files(folder, base):
    files = glob(os.path.join(folder, base+'*'))
    for fname in files:
        basename = os.path.basename(fname)
        _, ext = os.path.splitext(basename)
        os.rename(os.path.join(folder, basename),
                  os.path.join(folder, base + ext))


def process_peter(videos,
                  model_folder,
                  out_folder,
                  video_type='avi'):
    os.makedirs(out_folder, exist_ok=True)
    config_name = os.path.join(model_folder, 'config.yaml')

    import deeplabcut
    trap = io.StringIO()
    for i in range(0, len(videos), 5):
        batch = videos[i:i + 5]
        for video in batch:
            print(video)
        with redirect_stdout(trap):
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            deeplabcut.analyze_videos(config_name,
                                      batch,
                                      videotype=video_type,
                                      save_as_csv=True,
                                      destfolder=out_folder,
                                      TFGPUinference=True)
        for video in batch:
            basename = os.path.basename(video)
            basename, ext = os.path.splitext(basename)
            rename_dlc_files(out_folder, basename)