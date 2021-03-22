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
import numpy as np
import pandas as pd
import shutil

testing_directory = r'C:\Users\Peter\Desktop\DEG\mouse_reach_deepethogram'
data_directory = os.path.join(testing_directory, 'DATA')
config_path = os.path.join(testing_directory, 'project_config.yaml')

BATCH_SIZE = 16 # small but not too small
STEPS_PER_EPOCH = 500 # if less than 10, might have bugs with visualization
NUM_EPOCHS = 500

def command_from_string(string):
    command = string.split(' ')
    if command[-1] == '':
        command = command[:-1]
    return command

def add_default_arguments(string, train=True):
    string += f'project.config_file={config_path} '
    string += f'compute.batch_size={BATCH_SIZE} '
    if train:
        string += f'train.steps_per_epoch.train={STEPS_PER_EPOCH} '
        string += f'train.steps_per_epoch.val={STEPS_PER_EPOCH} '
        string += f'train.steps_per_epoch.test={20} '
        string += f'train.num_epochs={NUM_EPOCHS} '
    return string

# TRAINING

## pretrained weights
# pretrained_folder = r'C:\Users\Peter\Desktop\DEG\mouse_reach_deepethogram\models\pretrained'
# pretrained_flow_weights = os.path.join(
#     pretrained_folder,
#     '200221_115158_TinyMotionNet',
#     'checkpoint.pt')
#
# pretrained_feature_weights = os.path.join(
#     pretrained_folder,
#     '200415_125824_hidden_two_stream_kinetics_degf',
#     'checkpoint.pt')
#
# # train flow generation
# string = (f'python -m deepethogram.flow_generator.train '
#           f'preset=deg_f '
#           f'flow_generator.weights={pretrained_flow_weights} ')
# string = add_default_arguments(string)
# command = command_from_string(string)
# ret = subprocess.run(command)
#
# # train feature extraction
# string = (
#     f'python -m deepethogram.feature_extractor.train '
#     f'preset=deg_f '
#     f'flow_generator.weights=latest '
#     f'feature_extractor.weights={pretrained_feature_weights} ')
# string = add_default_arguments(string)
# command = command_from_string(string)
# ret = subprocess.run(command)
#
# # train sequence
# string = (f'python -m deepethogram.sequence.train ')
# string = add_default_arguments(string)
# command = command_from_string(string)
# print(command)
# ret = subprocess.run(command)

# # ANALYSIS
#
# Add videos to project
cfg = projects.load_config(config_path)
input_folder = r'C:\Users\Peter\Desktop\DATA\M5\2021_03_10'
video_dict = {x: tools.get_files(input_folder, (x + '.avi'))
              for x in ['CAM0']}
videos = video_dict['CAM0'][:2]
for video in videos:
    basename = os.path.basename(video)
    vidname = os.path.splitext(basename)[0]
    try:
        projects.add_video_to_project(cfg, video, mode='copy')
    except:
        print(f'{vidname} already exists')

subdirs = utils.get_subfiles(data_directory, 'directory')
dir_string = ','.join([str(i) for i in subdirs])
dir_string = '[' + dir_string + ']'

# feature extraction inference
string = (f'python -m deepethogram.feature_extractor.inference preset=deg_f reload.latest=True ')
string += f'inference.directory_list={dir_string} inference.overwrite=True '
string = add_default_arguments(string, train=False)
command = command_from_string(string)
ret = subprocess.run(command)

# sequential model extraction
string = (f'python -m deepethogram.sequence.inference preset=deg_f reload.latest=True ')
string += f'inference.directory_list={dir_string} inference.overwrite=True '
string = add_default_arguments(string, train=False)
command = command_from_string(string)
ret = subprocess.run(command)

# set predictions as labels with slight modification for grab probs.
np.set_printoptions(suppress=True)
for video in videos:
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
    thresholds[grab_ix] = 0.95
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

# FINAL
all_avis = tools.get_files(data_directory, ('.avi'))
for avi in all_avis:
    os.remove(avi)
save_dir = os.path.join(input_folder, 'ANALYZED', 'DEG')
os.makedirs(save_dir, exist_ok=True)

input_folder = r'C:\Users\Peter\Desktop\DATA\M5\2021_03_10'
for video in videos:
    vidname = os.path.basename(video)
    basename = os.path.splitext(vidname)[0]
    folder = os.path.join(data_directory, basename)
    save_folder = os.path.join(save_dir, basename)
    shutil.move(folder, save_folder)
    print(f'moved to {save_folder}')


# TODO: min bout length (small), interreach interval (long),


# # GUI
# string = (f'python -m deepethogram ')
# string += f'project={config_path} '
# command = command_from_string(string)
# ret = subprocess.run(command)

