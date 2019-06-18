# Copyright 2019 Sean Abreau and Antony Sagayaraj. All Rights Reserved.
import math
import os
import random
import tarfile
import urllib

from absl import app
from absl import flags
import tensorflow as tf

#from google.cloud import storage

# python imagenet_to_gcs.py \
#   --local_scratch_dir="./imagenet" \
#   --imagenet_username=asagayar \
#   --imagenet_access_key=3170eff67b6db60088681ccd516652298eb7cceb \

#flags.DEFINE_string(
#    'project', None, 'Google cloud project id for uploading the dataset.')
#flags.DEFINE_string(
#    'gcs_output_path', None, 'GCS path for uploading the dataset.')
flags.DEFINE_string(
    'local_scratch_dir', None, 'Scratch directory path for temporary files.')
flags.DEFINE_string(
    'raw_data_dir', None, 'Directory path for raw Imagenet dataset. '
    'Should have train and validation subdirectories inside it.')
flags.DEFINE_string(
    'imagenet_username', None, 'Username for Imagenet.org account')
flags.DEFINE_string(
    'imagenet_access_key', None, 'Access Key for Imagenet.org account')
#flags.DEFINE_boolean(
#    'gcs_upload', True, 'Set to false to not upload to gcs.')

FLAGS = flags.FLAGS

BASE_URL = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/'
LABELS_URL = 'https://raw.githubusercontent.com/tensorflow/models/master/research/inception/inception/data/imagenet_2012_validation_synset_labels.txt'  # pylint: disable=line-too-long

TRAINING_FILE = 'ILSVRC2012_img_train.tar'
VALIDATION_FILE = 'ILSVRC2012_img_val.tar'
LABELS_FILE = 'synset_labels.txt'

TRAINING_SHARDS = 1024
VALIDATION_SHARDS = 128

TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'
