import os
import subprocess
from deepethogram import utils
from deepethogram import projects
import deepethogram.flow_generator.train as flow_train
import deepethogram.feature_extractor.train as feature_train
import deepethogram.feature_extractor.inference
import deepethogram.sequence
import deepethogram.sequence.inference
import deepethogram.postprocessing as postpro
import tools
import video_tools
import numpy as np
import pandas as pd
import shutil
import pprint

BATCH_SIZE = 32
STEPS_PER_EPOCH = 500
NUM_EPOCHS = 500


def _command_from_string(string):
    command = string.split(' ')
    if command[-1] == '':
        command = command[:-1]
    return command

def _add_default_arguments(config_path, string, train=True):
    string += f'project.config_file={config_path} '
    string += f'compute.batch_size={BATCH_SIZE} '
    if train:
        string += f'train.steps_per_epoch.train={STEPS_PER_EPOCH} '
        string += f'train.steps_per_epoch.val={STEPS_PER_EPOCH} '
        string += f'train.steps_per_epoch.test={20} '
        string += f'train.num_epochs={NUM_EPOCHS} '
    return string

def train_deg(deg_directory, config_path):
    # pretrained weights
    pretrained_folder = os.path.join(deg_directory, 'models', 'pretrained')
    pretrained_flow_weights = os.path.join(
        pretrained_folder,
        '200221_115158_TinyMotionNet',
        'checkpoint.pt')

    pretrained_feature_weights = os.path.join(
        pretrained_folder,
        '200415_125824_hidden_two_stream_kinetics_degf',
        'checkpoint.pt')

    # # train flow generation
    string = (f'python -m deepethogram.flow_generator.train '
              f'preset=deg_f '
              f'flow_generator.weights={pretrained_flow_weights} ')
    string = _add_default_arguments(config_path, string)
    command = _command_from_string(string)
    ret = subprocess.run(command)

    # train feature extraction
    string = (
        f'python -m deepethogram.feature_extractor.train '
        f'preset=deg_f '
        f'flow_generator.weights=latest '
        f'feature_extractor.weights={pretrained_feature_weights} ')
    string = _add_default_arguments(config_path, string)
    command = _command_from_string(string)
    ret = subprocess.run(command)

    # train sequence
    string = (f'python -m deepethogram.sequence.train ')
    string = _add_default_arguments(config_path, string)
    command = _command_from_string(string)
    print(command)
    ret = subprocess.run(command)


def preprocess(input_folder, test):
    videos0 = [os.path.join(input_folder, x) for x in os.listdir(input_folder)
               if 'CAM0' in x]
    videos0 = sorted(videos0)
    videos1 = [os.path.join(input_folder, x) for x in os.listdir(input_folder)
               if 'CAM1' in x]
    videos1 = sorted(videos1)

    if test:
        videos0 = videos0[:2]
        videos1 = videos1[:2]

    collage_folder = os.path.join(input_folder, 'ANALYZED', 'COLLAGE')
    video_tools.make_collage(videos0,
                             width=5,
                             height=4,
                             collage_folder=collage_folder,
                             savestr='CAM0')
    merged_folder = os.path.join(input_folder, 'ANALYZED', 'MERGED')
    video_tools.merge_cams(videos0, videos1, merged_folder)


def analyze(input_folder, deg_directory, name, test):
    data_directory = os.path.join(deg_directory, 'DATA')
    config_path = os.path.join(deg_directory, 'project_config.yaml')
    cfg = projects.load_config(config_path)
    merged_folder = os.path.join(input_folder, 'ANALYZED', 'MERGED')
    videos = [os.path.join(merged_folder, x) for x in os.listdir(merged_folder) if
              '.mp4' in x]

    if test:
        videos = videos[:2]

    # Add videos to project
    # shutil.rmtree(data_directory)
    # os.makedirs(data_directory)
    for video in videos:
        basename = os.path.basename(video)
        vidname = os.path.splitext(basename)[0]
        try:
            projects.add_video_to_project(cfg, video, mode='copy')
        except:
            print(f'{vidname} already exists')

    subdirs = [os.path.join(data_directory, x) for x in os.listdir(
        data_directory) if name in x]
    dir_string = ','.join([str(i) for i in subdirs])
    dir_string = '[' + dir_string + ']'

    # feature extraction inference
    string = (f'python -m deepethogram.feature_extractor.inference preset=deg_f reload.latest=True ')
    string += f'inference.directory_list={dir_string} inference.overwrite=True '
    string = _add_default_arguments(config_path, string, train=False)
    command = _command_from_string(string)
    ret = subprocess.run(command)
    #
    # sequential model extraction
    string = (f'python -m deepethogram.sequence.inference preset=deg_f reload.latest=True ')
    string += f'inference.directory_list={dir_string} inference.overwrite=True '
    string = _add_default_arguments(config_path, string, train=False)
    command = _command_from_string(string)
    ret = subprocess.run(command)

    # set predictions as labels with slight modification for probs.
    np.set_printoptions(suppress=True)
    for i, video in enumerate(videos):
        vidname = os.path.basename(video)
        basename = os.path.splitext(vidname)[0]
        outputfile = os.path.join(data_directory,
                                  basename,
                                  basename + '_outputs.h5')
        outputs = projects.import_outputfile(cfg['project']['path'],
                                             outputfile,
                                             class_names=cfg['project']['class_names'],
                                             latent_name=None)

        probabilities, thresholds, latent_name, keys = outputs
        grab_ix = cfg['project']['class_names'].index('grab')
        drop_ix = cfg['project']['class_names'].index('dropped')
        # thresholds[grab_ix] = 0.95
        thresholds[drop_ix] = 0.5
        postprocessor = postpro.MinBoutLengthPostprocessor(thresholds,
                                                           bout_length=2)
        estimated_labels = postprocessor(probabilities)

        df = pd.DataFrame(data=estimated_labels,
                          columns=cfg['project']['class_names'])
        prediction_fname = os.path.join(data_directory,
                                        basename,
                                        basename + '_labels.csv')
        df.to_csv(prediction_fname)
        print(df.sum(axis=0))
        print(prediction_fname)
        print(thresholds)

    # move analyzed data to the local data directory and delete the avi files
    save_dir = os.path.join(input_folder, 'ANALYZED', 'LABELS')
    os.makedirs(save_dir, exist_ok=True)
    for video in videos:
        vidname = os.path.basename(video)
        basename = os.path.splitext(vidname)[0]
        folder = os.path.join(data_directory, basename)
        save_folder = os.path.join(save_dir, basename)
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        shutil.copytree(folder, save_folder)

if __name__ == '__main__':
    mouse_dates = [
        # r'M5\2021_03_06',
        # r'M5\2021_03_07',
        r'M5\2021_03_08',
        r'M9\2021.03.05',
        r'M9\2021.03.06',
        r'M9\2021.03.07',
        r'M9\2021.03.08',
        r'M9\2021.03.09',
        r'M9\2021.03.10',
        r'M9\2021.03.11',
        r'M9\2021.03.12',
        r'M9\2021.03.14',
    ]
    test = False
    for mouse_date in mouse_dates:
        name = mouse_date.replace('\\', '_')
        desktop_directory = os.path.join(r'C:\Users',
                                         os.getlogin(),
                                         'Desktop')
        input_folder = os.path.join(desktop_directory, 'DATA', mouse_date)
        deg_directory = os.path.join(desktop_directory,
                                     r'DEG\reach_merged_v0_deepethogram')

        preprocess(input_folder, test=test)
        analyze(input_folder, deg_directory, name, test=test)
