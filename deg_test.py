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
from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array
import pprint


deg_directory = r'C:\Users\Peter\Desktop\DEG\reach_merged_v0_deepethogram'
data_directory = os.path.join(deg_directory, 'DATA')
config_path = os.path.join(deg_directory, 'project_config.yaml')
cfg = projects.load_config(config_path)

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


# # PREPROCESSING
# input_folder = r'C:\Users\Peter\Desktop\DATA\M5\2021_03_11'
# videos0 = [os.path.join(input_folder, x) for x in os.listdir(input_folder)
#            if 'CAM0' in x]
# videos0 = sorted(videos0)
# videos1 = [os.path.join(input_folder, x) for x in os.listdir(input_folder)
#            if 'CAM1' in x]
# videos1 = sorted(videos1)
#
# merged_folder = os.path.join(input_folder, 'ANALYZED', 'MERGED')
# os.makedirs(merged_folder, exist_ok=True)
# for v0, v1 in zip(videos0, videos1):
#     basename = os.path.basename(v0)
#     vidname = os.path.splitext(basename)[0]
#     pn = os.path.join(merged_folder, vidname[:-4] + '.mp4')
#
#     if not os.path.exists(pn):
#         clip1 = VideoFileClip(v0)
#         clip2 = VideoFileClip(v1)
#         final_clip = clips_array([[clip1, clip2]])
#         final_clip.write_videofile(pn, codec='libx264')
#
# videos = [os.path.join(merged_folder, x) for x in os.listdir(merged_folder) if
#           '.mp4' in x]
# # Add videos to project
# for video in videos:
#     basename = os.path.basename(video)
#     vidname = os.path.splitext(basename)[0]
#     try:
#         projects.add_video_to_project(cfg, video, mode='copy')
#     except:
#         print(f'{vidname} already exists')

# TRAINING

# pretrained weights
pretrained_folder = r'C:\Users\Peter\Desktop\DEG\mouse_reach_deepethogram\models\pretrained'
pretrained_flow_weights = os.path.join(
    pretrained_folder,
    '200221_115158_TinyMotionNet',
    'checkpoint.pt')

pretrained_feature_weights = os.path.join(
    pretrained_folder,
    '200415_125824_hidden_two_stream_kinetics_degf',
    'checkpoint.pt')

# train flow generation
string = (f'python -m deepethogram.flow_generator.train '
          f'preset=deg_f '
          f'flow_generator.weights={pretrained_flow_weights} ')
string = add_default_arguments(string)
command = command_from_string(string)
ret = subprocess.run(command)

# train feature extraction
string = (
    f'python -m deepethogram.feature_extractor.train '
    f'preset=deg_f '
    f'flow_generator.weights=latest '
    f'feature_extractor.weights={pretrained_feature_weights} ')
string = add_default_arguments(string)
command = command_from_string(string)
ret = subprocess.run(command)

# train sequence
string = (f'python -m deepethogram.sequence.train ')
string = add_default_arguments(string)
command = command_from_string(string)
print(command)
ret = subprocess.run(command)

# # # ANALYSIS
# subdirs = utils.get_subfiles(data_directory, 'directory')
# dir_string = ','.join([str(i) for i in subdirs])
# dir_string = '[' + dir_string + ']'
#
# # feature extraction inference
# string = (f'python -m deepethogram.feature_extractor.inference preset=deg_f reload.latest=True ')
# string += f'inference.directory_list={dir_string} inference.overwrite=True '
# string = add_default_arguments(string, train=False)
# command = command_from_string(string)
# ret = subprocess.run(command)
#
# # sequential model extraction
# string = (f'python -m deepethogram.sequence.inference preset=deg_f reload.latest=True ')
# string += f'inference.directory_list={dir_string} inference.overwrite=True '
# string = add_default_arguments(string, train=False)
# command = command_from_string(string)
# ret = subprocess.run(command)
#
# # set predictions as labels with slight modification for grab probs.
# np.set_printoptions(suppress=True)
# for video in videos:
#     vidname = os.path.basename(video)
#     basename = os.path.splitext(vidname)[0]
#     outputfile = os.path.join(data_directory,
#                               basename,
#                               basename + '_outputs.h5')
#     outputs = projects.import_outputfile(cfg['project']['path'],
#                                          outputfile,
#                                          class_names=cfg['project']['class_names'],
#                                          latent_name=None)
#
#     probabilities, thresholds, latent_name, keys = outputs
#     grab_ix = cfg['project']['class_names'].index('grab')
#     thresholds[grab_ix] = 0.95
#     postprocessor = postpro.MinBoutLengthPostprocessor(thresholds,
#                                                        bout_length=2)
#     estimated_labels = postprocessor(probabilities)
#
#     df = pd.DataFrame(data=estimated_labels,
#                       columns=cfg['project']['class_names'])
#     prediction_fname = os.path.join(data_directory,
#                                     basename,
#                                     basename + '_labels.csv')
#     df.to_csv(prediction_fname)
#     print(df.sum(axis=0))
#     print(prediction_fname)
# #
# # FINAL
# save_dir = os.path.join(input_folder, 'ANALYZED', 'LABELS')
# os.makedirs(save_dir, exist_ok=True)
#
# # move analyzed data to the local data directory and delete the avi files
# for video in videos:
#     vidname = os.path.basename(video)
#     basename = os.path.splitext(vidname)[0]
#     folder = os.path.join(data_directory, basename)
#     save_folder = os.path.join(save_dir, basename)
#     shutil.copytree(folder, save_folder)
#     print(f'moved to {save_folder}')
# all_avis = tools.get_files(save_dir, ('.avi'))
# for avi in all_avis:
#     os.remove(avi)




# TODO: min bout length (small), interreach interval (long),


# # GUI
# string = (f'python -m deepethogram ')
# string += f'project={config_path} '
# command = command_from_string(string)
# ret = subprocess.run(command)

