import os
import subprocess
from deepethogram import utils
import deepethogram.flow_generator.train as flow_train
import deepethogram.feature_extractor.train as feature_train

d = r'C:\Users\Peter\Desktop\DEG\mouse_reach_deepethogram\DATA\M9_2021.03' \
    r'.08_00000_FRONT\M9_2021.03.08_00000_FRONT_outputs.h5'
testing_directory = r'C:\Users\Peter\Desktop\DEG\mouse_reach_deepethogram'
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
          f'flow_generator.weights=latest ')
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

# test feature extraction
string = (f'python -m deepethogram.feature_extractor.inference preset=deg_f reload.latest=True ')
datadir = os.path.join(testing_directory, 'DATA')
subdirs = utils.get_subfiles(datadir, 'directory')
dir_string = ','.join([str(i) for i in subdirs])
dir_string = '[' + dir_string + ']'
string += f'inference.directory_list={dir_string} inference.overwrite=True '
string = add_default_arguments(string, train=False)
command = command_from_string(string)
ret = subprocess.run(command)
#
# train sequence
string = (f'python -m deepethogram.sequence.train ')
string = add_default_arguments(string)
command = command_from_string(string)
print(command)
ret = subprocess.run(command)