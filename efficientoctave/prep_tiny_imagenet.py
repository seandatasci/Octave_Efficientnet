# Copyright 2019 seandatasci and Antony Sagayaraj. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
#   --raw_data_dir="./imagenet/raw_data" \
#   --imagenet_username=asagayar \
#   --imagenet_access_key=3170eff67b6db60088681ccd516652298eb7cceb \



import os
import subprocess

label_dirs = os.listdir()

for label in label_dirs:
  bashCommand = "mv " + label + "/images ../new/" + label
  process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
  output, error = process.communicate()

# import re

# idp = re.compile('n[0-9]{8}')

# synset_labels = []
# lines = open('val_annotations.txt','r').readlines()
# for item in lines:
#   x = re.search(idp,item)
#   synset_labels.append(x.group())

# outfile = open('synset_labels.txt','w')
# outfile.write("\n".join(synset_labels))
# outfile.close()

# TRAINING_FILE = 'ILSVRC2012_img_train.tar'
# VALIDATION_FILE = 'ILSVRC2012_img_val.tar'
LABELS_FILE = 'synset_labels.txt'

TRAINING_SHARDS = 1024
VALIDATION_SHARDS = 128

TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'

####
def _check_or_create_dir(directory):
  """Check if directory exists otherwise create it."""
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  # new_value = bytes(value,'utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, synset, height, width):
  """Build an Example proto for an example.
  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  colorspace = bytes('RGB','utf-8')
  channels = 3
  image_format = bytes('JPEG','utf-8')
  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/synset': _bytes_feature(bytes(synset,'utf-8')),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(bytes(os.path.basename(filename),'utf-8')),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example


def _is_png(filename):
  """Determine if a file contains a PNG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a PNG.
  """
  # File list from:
  # https://github.com/cytsai/ilsvrc-cmyk-image-list
  return 'n02105855_2933.JPEG' in filename


def _is_cmyk(filename):
  """Determine if file contains a CMYK JPEG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a JPEG encoded with CMYK color space.
  """
  # File list from:
  # https://github.com/cytsai/ilsvrc-cmyk-image-list
  blacklist = set(['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
                   'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
                   'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
                   'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
                   'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
                   'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
                   'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
                   'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
                   'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
                   'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
                   'n07583066_647.JPEG', 'n13037406_4650.JPEG'])
  return os.path.basename(filename) in blacklist


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""
  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()
    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})
  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})
  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _process_image(filename, coder):
  """Process a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()
  # Clean the dirty data.
  if _is_png(filename):
    # 1 image is a PNG.
    tf.logging.info('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)
  elif _is_cmyk(filename):
    # 22 JPEG images are in CMYK colorspace.
    tf.logging.info('Converting CMYK to RGB for %s' % filename)
    image_data = coder.cmyk_to_rgb(image_data)
  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)
  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3
  return image_data, height, width


def _process_image_files_batch(coder, output_file, filenames, synsets, labels):
  """Processes and saves list of images as TFRecords.
  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    output_file: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: map of string to integer; id for all synset labels
  """
  writer = tf.python_io.TFRecordWriter(output_file)
  for filename, synset in zip(filenames, synsets):
    image_buffer, height, width = _process_image(filename, coder)
    label = labels[synset]
    example = _convert_to_example(filename, image_buffer, label,
                                  synset, height, width)
    writer.write(example.SerializeToString())
  writer.close()


def _process_dataset(filenames, synsets, labels, output_directory, prefix,
                     num_shards):
  """Processes and saves list of images as TFRecords.
  Args:
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: map of string to integer; id for all synset labels
    output_directory: path where output files should be created
    prefix: string; prefix for each file
    num_shards: number of chucks to split the filenames into
  Returns:
    files: list of tf-record filepaths created from processing the dataset.
  """
  _check_or_create_dir(output_directory)
  chunksize = int(math.ceil(len(filenames) / num_shards))
  coder = ImageCoder()
  files = []
  for shard in range(num_shards):
    chunk_files = filenames[shard * chunksize : (shard + 1) * chunksize]
    chunk_synsets = synsets[shard * chunksize : (shard + 1) * chunksize]
    output_file = os.path.join(
        output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, num_shards))
    _process_image_files_batch(coder, output_file, chunk_files,
                               chunk_synsets, labels)
    tf.logging.info('Finished writing file: %s' % output_file)
    files.append(output_file)
  return files


def convert_to_tf_records(raw_data_dir):
  """Convert the Imagenet dataset into TF-Record dumps."""

  # Shuffle training records to ensure we are distributing classes
  # across the batches.
  random.seed(0)
  def make_shuffle_idx(n):
    order = list(range(n))
    random.shuffle(order)
    return order

  # Glob all the training files
  training_files = tf.gfile.Glob(
      os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*', '*.JPEG'))

  # Get training file synset labels from the directory name
  training_synsets = [
      os.path.basename(os.path.dirname(f)) for f in training_files]

  training_shuffle_idx = make_shuffle_idx(len(training_files))
  training_files = [training_files[i] for i in training_shuffle_idx]
  training_synsets = [training_synsets[i] for i in training_shuffle_idx]

  # Glob all the validation files
  validation_files = sorted(tf.gfile.Glob(
      os.path.join(raw_data_dir, VALIDATION_DIRECTORY, '*.JPEG')))

  # Get validation file synset labels from labels.txt
  validation_synsets = tf.gfile.FastGFile(
      os.path.join(raw_data_dir, LABELS_FILE), 'r').read().splitlines()

  # Create unique ids for all synsets
  labels = {v: k + 1 for k, v in enumerate(
      sorted(set(validation_synsets + training_synsets)))}

  # Create training data
  tf.logging.info('Processing the training data.')
  training_records = _process_dataset(
      training_files, training_synsets, labels,
      os.path.join(local_scratch_dir, TRAINING_DIRECTORY),
      TRAINING_DIRECTORY, TRAINING_SHARDS)




  # Create validation data
  tf.logging.info('Processing the validation data.')
  validation_records = _process_dataset(
      validation_files, validation_synsets, labels,
      os.path.join(local_scratch_dir, VALIDATION_DIRECTORY),
      VALIDATION_DIRECTORY, VALIDATION_SHARDS)

  return training_records, validation_records


local_scratch_dir = "./tiny-imagenet"
raw_data_dir = "./tiny-imagenet/raw_data"
#training_records, validation_records = convert_to_tf_records(raw_data_dir)
random.seed(0)
def make_shuffle_idx(n):
    order = list(range(n))
    random.shuffle(order)
    return order

# Glob all the training files
training_files = tf.gfile.Glob(
    os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*', '*.JPEG'))

# Get training file synset labels from the directory name
training_synsets = [
    os.path.basename(os.path.dirname(f)) for f in training_files]

training_shuffle_idx = make_shuffle_idx(len(training_files))
training_files = [training_files[i] for i in training_shuffle_idx]
training_synsets = [training_synsets[i] for i in training_shuffle_idx]

# Glob all the validation files
validation_files = sorted(tf.gfile.Glob(
    os.path.join(raw_data_dir, VALIDATION_DIRECTORY, '*.JPEG')))

# Get validation file synset labels from labels.txt
validation_synsets = tf.gfile.FastGFile(
    os.path.join(raw_data_dir, LABELS_FILE), 'r').read().splitlines()

# Create unique ids for all synsets
labels = {v: k + 1 for k, v in enumerate(
    sorted(set(validation_synsets + training_synsets)))}

# Create training data
tf.logging.info('Processing the training data.')
training_records = _process_dataset(
    training_files, training_synsets, labels,
    os.path.join(local_scratch_dir, TRAINING_DIRECTORY),
    TRAINING_DIRECTORY, TRAINING_SHARDS)

# Create validation data
tf.logging.info('Processing the validation data.')
validation_records = _process_dataset(
    validation_files, validation_synsets, labels,
    os.path.join(local_scratch_dir, VALIDATION_DIRECTORY),
    VALIDATION_DIRECTORY, VALIDATION_SHARDS)
