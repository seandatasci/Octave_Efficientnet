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
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import utils

from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
#activation function
relu_fn = tf.nn.swish

#octave convolution class including depthwise
class OctConv2D(layers.Layer):
  def __init__(self, filters, alpha, kernel_size=(3, 3), strides=(1, 1),
                padding="same", kernel_initializer='conv_kernel_initializer',
                kernel_regularizer=None, kernel_constraint=None,
                use_depthwise=False, use_bias=False,**kwargs):
    """
    OctConv2D : Octave Convolution for image( rank 4 tensors)
    filters: # output channels for low + high
    alpha: Low channel ratio (alpha=0 -> High only, alpha=1 -> Low only)
    kernel_size : 3x3 by default, padding : same by default
    """
    assert alpha >= 0 and alpha <= 1
    assert filters > 0 and isinstance(filters, int)
    super().__init__(**kwargs)

    self.alpha = alpha
    self.filters = filters
    # optional values
    self.kernel_size = kernel_size
    self.strides = strides
    if strides == (2,2) or strides == [2,2]:
      self.strided=True
    else:
      self.strided=False
    self.padding = padding
    self.kernel_initializer = kernel_initializer
    self.kernel_regularizer = kernel_regularizer
    self.kernel_constraint = kernel_constraint
    self.use_depthwise = use_depthwise
    self.use_bias = use_bias
    # -> Low Channels
    self.low_channels = int(self.filters * self.alpha)
    # -> High Channles
    self.high_channels = self.filters - self.low_channels

  def build(self, input_shape):
    assert len(input_shape) == 2
    assert len(input_shape[0]) == 4 and len(input_shape[1]) == 4
    # Assertion for high inputs
    assert input_shape[0][1] // 2 >= self.kernel_size[0]
    assert input_shape[0][2] // 2 >= self.kernel_size[1]
    # Assertion for low inputs
    assert input_shape[0][1] // input_shape[1][1] == 2
    assert input_shape[0][2] // input_shape[1][2] == 2
    # channels last for TensorFlow
    assert K.image_data_format() == "channels_last"
    # input channels
    high_in = int(input_shape[0][3])
    low_in = int(input_shape[1][3])

    # High -> High conv
    if self.use_depthwise:
      self.high_to_high = utils.DepthwiseConv2D(self.kernel_size,
                                                strides=(1, 1),
                                                depthwise_initializer=self.kernel_initializer,
                                                padding=self.padding,
                                                use_bias=self.use_bias)
      # Low -> Low conv
      self.low_to_low = utils.DepthwiseConv2D(self.kernel_size,
                                              strides=(1, 1),
                                              depthwise_initializer=self.kernel_initializer,
                                              padding=self.padding,
                                              use_bias=self.use_bias)
      self.high_to_low = None
      self.low_to_high = None
    else:
      self.high_to_high = tf.layers.Conv2D(self.high_channels,
                                            kernel_size=self.kernel_size,
                                            strides=(1, 1),
                                            kernel_initializer=self.kernel_initializer,
                                            padding=self.padding,
                                            use_bias=self.use_bias)
      # Low -> Low conv
      self.low_to_low = tf.layers.Conv2D(self.low_channels,
                                          kernel_size=self.kernel_size,
                                          strides=(1, 1),
                                          kernel_initializer=self.kernel_initializer,
                                          padding=self.padding,
                                          use_bias=False)                                   
      # High -> Low conv
      self.high_to_low = tf.layers.Conv2D(self.low_channels,
                                          kernel_size=self.kernel_size,
                                          strides=(1, 1),
                                          kernel_initializer=self.kernel_initializer,
                                          padding=self.padding,
                                          use_bias=self.use_bias)
      # Low -> High conv
      self.low_to_high = tf.layers.Conv2D(self.high_channels,
                                          kernel_size=self.kernel_size,
                                          strides=(1, 1),
                                          kernel_initializer=self.kernel_initializer,
                                          padding=self.padding,
                                          use_bias=False)
    super().build(input_shape)
  
  def call(self, inputs):
    # Input = [X^H, X^L]
    assert len(inputs) == 2
    high_input, low_input = inputs

    if self.use_depthwise:         
      if self.strided:
        h2h_input = layers.AveragePooling2D(2)(high_input)
        l2l_input = layers.AveragePooling2D(2)(low_input)
      else:
        h2h_input = high_input
        l2l_input = low_input
      # High -> High conv
      high = self.high_to_high(h2h_input)
      # Low -> Low conv
      low = self.low_to_low(l2l_input)
      # Low -> High conv
      high_from_low = None
      # High -> Low conv
      low_from_high = None
    else:
      if self.strided:
        h2h_input = layers.AveragePooling2D(2)(high_input)
        l2l_input = layers.AveragePooling2D(2)(low_input)
      else:
        h2h_input = high_input
        l2l_input = low_input
      # High -> High conv
      high = self.high_to_high(h2h_input)
      # Low -> Low conv
      low = self.low_to_low(l2l_input)
      # Low -> High conv
      high_from_low = self.low_to_high(low_input)
      if not self.strided:
        high_from_low = layers.UpSampling2D(size=(2, 2))(high_from_low)
      # High -> Low conv
      low_from_high = layers.AveragePooling2D(2)(h2h_input)
      low_from_high = self.high_to_low(low_from_high)
    # Cross Add
    if self.use_depthwise:
      high_add = high
      low_add = low
    else:
      high_add = high + high_from_low
      low_add = low + low_from_high
    return [high_add, low_add]

  def compute_output_shape(self, input_shapes):
    high_in_shape, low_in_shape = input_shapes
    high_out_shape = (*high_in_shape[:3], self.high_channels)
    low_out_shape = (*low_in_shape[:3], self.low_channels)
    return [high_out_shape, low_out_shape]

  def get_config(self):
    base_config = super().get_config()
    out_config = {
        **base_config,
        "alpha": self.alpha,
        "filters": self.filters,
        "kernel_size": self.kernel_size,
        "strides": self.strides,
        "padding": self.padding,
        "kernel_initializer": self.kernel_initializer,
        "kernel_regularizer": self.kernel_regularizer,
        "kernel_constraint": self.kernel_constraint,
    }
    return out_config

    

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

#batchnorm = tf.layers.BatchNormalization
batchnorm = utils.TpuBatchNormalization  # TPU-specific requirement.


BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for convolutional kernels.
  The main difference with tf.variance_scaling_initializer is that
  tf.variance_scaling_initializer uses a truncated normal with an uncorrected
  standard deviation, whereas here we use a normal distribution. Similarly,
  tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
  a corrected standard deviation.
  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused
  Returns:
    an initialization for the variable
  """
  del partition_info
  kernel_height, kernel_width, _, out_filters = shape
  fan_out = int(kernel_height * kernel_width * out_filters)
  return tf.random_normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for dense kernels.
  This initialization is equal to
    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                    distribution='uniform').
  It is written out explicitly here for clarity.
  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused
  Returns:
    an initialization for the variable
  """
  del partition_info
  init_range = 1.0 / np.sqrt(shape[1])
  return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


def round_filters(filters, global_params):
  """Round number of filters based on depth multiplier."""
  orig_f = filters
  multiplier = global_params.width_coefficient
  divisor = global_params.depth_divisor
  min_depth = global_params.min_depth
  if not multiplier:
    return filters

  filters *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  tf.logging.info('round_filter input={} output={}'.format(orig_f, new_filters))
  return int(new_filters)


def round_repeats(repeats, global_params):
  """Round number of filters based on depth multiplier."""
  multiplier = global_params.depth_coefficient
  if not multiplier:
    return repeats
  return int(math.ceil(multiplier * repeats))


class MBConvBlock(object):
  """A class of MBConv: Mobile Inverted Residual Bottleneck.
  Attributes:
    has_se: boolean. Whether the block contains a Squeeze and Excitation layer
      inside.
    endpoints: dict. A list of internal tensors.
  """

  def __init__(self, block_args, global_params):
    """Initializes a MBConv block.
    Args:
      block_args: BlockArgs, arguments to create a Block.
      global_params: GlobalParams, a set of global parameters.
    """
    self._block_args = block_args
    self._batch_norm_momentum = global_params.batch_norm_momentum
    self._batch_norm_epsilon = global_params.batch_norm_epsilon
    if global_params.data_format == 'channels_first':
      self._channel_axis = 1
      self._spatial_dims = [2, 3]
    else:
      self._channel_axis = -1
      self._spatial_dims = [1, 2]
    self.has_se = (self._block_args.se_ratio is not None) and (
        self._block_args.se_ratio > 0) and (self._block_args.se_ratio <= 1)

    self.endpoints = None

    # Builds the block accordings to arguments.
    self._build()

  def block_args(self):
    return self._block_args

  def _build(self):
    """Builds block according to the arguments."""
    filters = self._block_args.input_filters * self._block_args.expand_ratio
    if self._block_args.expand_ratio != 1:
      # Expansion phase:
      self._expand_conv = OctConv2D(
          filters,
          alpha=0.125,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=False)
      self._bn0_h = batchnorm(
          axis=self._channel_axis,
          momentum=self._batch_norm_momentum,
          epsilon=self._batch_norm_epsilon)
      self._bn0_l = batchnorm(
          axis=self._channel_axis,
          momentum=self._batch_norm_momentum,
          epsilon=self._batch_norm_epsilon)
      
    kernel_size = self._block_args.kernel_size
    
    # Depth-wise convolution phase:
    self._depthwise_conv = OctConv2D(
        filters,
        alpha=0.125,
        kernel_size=[kernel_size, kernel_size],
        strides=self._block_args.strides,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        use_depthwise=True)
    self._bn1_h = batchnorm(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon)
    self._bn1_l = batchnorm(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon)

    if self.has_se:
      num_reduced_filters = max(
          1, int(self._block_args.input_filters * self._block_args.se_ratio))
      # Squeeze and Excitation layer.
      self._se_reduce_h = tf.layers.Conv2D(
          num_reduced_filters,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=True)
      self._se_reduce_l = tf.layers.Conv2D(
          num_reduced_filters,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=True)
      if self._block_args.expand_ratio != 1:
        high_filters = filters-int(filters*0.125)
        low_filters = int(filters*0.125)
      else:
        high_filters = filters
        low_filters = filters
      self._se_expand_h = tf.layers.Conv2D(
          high_filters,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=True)
      self._se_expand_l = tf.layers.Conv2D(
          low_filters,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=True)

    # Output phase:
    filters = self._block_args.output_filters
    self._project_conv = OctConv2D(
        filters,
        alpha=0.125,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same')
    self._bn2_h = batchnorm(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon)
    self._bn2_l = batchnorm(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon)

  def _call_se(self, input_tensors):
    """Call Squeeze and Excitation layer.
    Args:
      input_tensors: High and Low tensors for Squeeze/Excitation layer.
    Returns:
      A output tensor, which should have the same shape as input.
    """
    input_high,input_low = input_tensors
    se_high = tf.reduce_mean(input_high, self._spatial_dims, keepdims=True)
    se_low = tf.reduce_mean(input_low, self._spatial_dims, keepdims=True)
    se_high = self._se_expand_h(relu_fn(self._se_reduce_h(se_high)))
    se_low = self._se_expand_l(relu_fn(self._se_reduce_l(se_low)))
    se_high = tf.sigmoid(se_high) * input_high
    se_low = tf.sigmoid(se_low) * input_low
    tf.logging.info('Built Squeeze and Excitation with tensor shape: %s' %
                    (se_high.shape))
    return se_high, se_low

  def call(self, inputs, training=True, drop_connect_rate=None):
    """Implementation of call().
    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      drop_connect_rate: float, between 0 to 1, drop connect rate.
    Returns:
      A output tensor.
    """
    high = inputs
    low = layers.AveragePooling2D(2)(inputs)

    tf.logging.info('Block input: %s shape: %s' % (high.name, high.shape))
    if self._block_args.expand_ratio != 1:
      high, low = self._expand_conv([inputs,low])
      high = relu_fn(self._bn0_h(high, training=training))
      low = relu_fn(self._bn0_l(low, training=training))
    else:
      pass

    tf.logging.info('Expand: %s shape: %s' % (high.name, high.shape))

    high, low = self._depthwise_conv([high,low])
    high = relu_fn(self._bn1_h(high, training=training))
    low = relu_fn(self._bn1_l(low, training=training))
    tf.logging.info('DWConv: %s shape: %s' % (high.name, high.shape))

    if self.has_se:
      with tf.variable_scope('se'):
        high,low = self._call_se([high,low])
        low = layers.UpSampling2D(size=(2, 2))(low)
        concat = layers.Concatenate()([high, low])
        high = inputs
        low = layers.AveragePooling2D(2)(inputs)        

    self.endpoints = {'expansion_output': high}
    high, low = self._project_conv([high,low])
    high = self._bn2_h(high, training=training)
    low = self._bn2_l(low, training=training)
    
    low = layers.UpSampling2D(size=(2, 2))(low)
    x = layers.Concatenate()([high, low])
    if self._block_args.id_skip:
      if all(
          s == 1 for s in self._block_args.strides
      ) and self._block_args.input_filters == self._block_args.output_filters:
        # only apply drop_connect if skip presents.
        if drop_connect_rate:
          x = utils.drop_connect(x, training, drop_connect_rate)
        x = tf.add(x, inputs)
    tf.logging.info('Project: %s shape: %s' % (x.name, x.shape))
    return x


class Model(tf.keras.Model):
  """A class implements tf.keras.Model for MNAS-like model.
    Reference: https://arxiv.org/abs/1807.11626
  """

  def __init__(self, blocks_args=None, global_params=None):
    """Initializes an `Model` instance.
    Args:
      blocks_args: A list of BlockArgs to construct block modules.
      global_params: GlobalParams, a set of global parameters.
    Raises:
      ValueError: when blocks_args is not specified as a list.
    """
    super(Model, self).__init__()
    if not isinstance(blocks_args, list):
      raise ValueError('blocks_args should be a list.')
    self._global_params = global_params
    self._blocks_args = blocks_args
    self.endpoints = None
    self._build()

  def _build(self):
    """Builds a model."""
    self._blocks = []
    # Builds blocks.
    for block_args in self._blocks_args:
      assert block_args.num_repeat > 0
      # Update block input and output filters based on depth multiplier.
      block_args = block_args._replace(
          input_filters=round_filters(block_args.input_filters,
                                      self._global_params),
          output_filters=round_filters(block_args.output_filters,
                                       self._global_params),
          num_repeat=round_repeats(block_args.num_repeat, self._global_params))

      # The first block needs to take care of stride and filter size increase.
      self._blocks.append(MBConvBlock(block_args, self._global_params))
      if block_args.num_repeat > 1:
        # pylint: disable=protected-access
        block_args = block_args._replace(
            input_filters=block_args.output_filters, strides=[1, 1])
        # pylint: enable=protected-access
      for _ in xrange(block_args.num_repeat - 1):
        self._blocks.append(MBConvBlock(block_args, self._global_params))

    batch_norm_momentum = self._global_params.batch_norm_momentum
    batch_norm_epsilon = self._global_params.batch_norm_epsilon
    if self._global_params.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1

    # Stem part.
    self._conv_stem = OctConv2D(
        filters=round_filters(32, self._global_params),
        alpha=0.125,
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False)
    self._bn0_h = batchnorm(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)
    self._bn0_l = batchnorm(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)

    # Head part.
    self._conv_head = OctConv2D(
        filters=round_filters(1280, self._global_params),
        alpha=0.125,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False)
    self._bn1_h = batchnorm(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)
    self._bn1_l = batchnorm(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)
    
    self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
        data_format=self._global_params.data_format)
    self._fc = tf.layers.Dense(
        self._global_params.num_classes,
        kernel_initializer=dense_kernel_initializer)
    self.up_sample_stem = layers.Conv2DTranspose(filters = self._conv_stem.low_channels,
                                            kernel_size = 1, strides = 2, padding='same',
                                                data_format="channels_last")(low)
    self.up_sample_head = layers.Conv2DTranspose(filters = self._conv_head.low_channels,
                                            kernel_size = 1, strides = 2, padding='same',
                                                data_format="channels_last")(low)
    if self._global_params.dropout_rate > 0:
      self._dropout = tf.keras.layers.Dropout(self._global_params.dropout_rate)
    else:
      self._dropout = None

  def call(self, inputs, training=True, features_only=None):
    """Implementation of call().
    Args:
      inputs: input tensors.
      training: boolean, whether the model is constructed for training.
      features_only: build the base feature network only.
    Returns:
      output tensors.
    """
    outputs = None
    self.endpoints = {}
    # Calls Stem layers
    with tf.variable_scope('stem'):
        low = layers.AveragePooling2D(2)(inputs)
        high, low = self._conv_stem([inputs, low])
        high = relu_fn(self._bn0_h(high, training=training))
        low = relu_fn(self._bn0_l(low, training=training))
        low = self.up_sample_stem
        
        # low = layers.Conv2DTranspose(filters = ,kernel_size = 1, strides = 2, padding='same')
        outputs = layers.Concatenate()([high, low])
    tf.logging.info('Built stem layers with output shape: %s' % outputs.shape)
    self.endpoints['stem'] = outputs

    # Calls blocks.
    reduction_idx = 0
    for idx, block in enumerate(self._blocks):
      is_reduction = False
      if ((idx == len(self._blocks) - 1) or
          self._blocks[idx + 1].block_args().strides[0] > 1):
        is_reduction = True
        reduction_idx += 1

      with tf.variable_scope('blocks_%s' % idx):
        drop_rate = self._global_params.drop_connect_rate
        if drop_rate:
          drop_rate *= float(idx) / len(self._blocks)
          tf.logging.info('block_%s drop_connect_rate: %s' % (idx, drop_rate))
        outputs = block.call(
            outputs, training=training, drop_connect_rate=drop_rate)
        self.endpoints['block_%s' % idx] = outputs
        if is_reduction:
          self.endpoints['reduction_%s' % reduction_idx] = outputs
        if block.endpoints:
          for k, v in six.iteritems(block.endpoints):
            self.endpoints['block_%s/%s' % (idx, k)] = v
            if is_reduction:
              self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
    self.endpoints['global_pool'] = outputs

    if not features_only:
      # Calls final layers and returns logits.
      with tf.variable_scope('head'):
        low = layers.AveragePooling2D(2)(outputs)
        high, low = self._conv_head([outputs, low])
        high = relu_fn(self._bn1_h(high, training=training))
        low = relu_fn(self._bn1_l(low, training=training))
        low = self.up_sample_head
        outputs = layers.Concatenate()([high, low])
        outputs = self._avg_pooling(outputs)
        if self._dropout:
          outputs = self._dropout(outputs, training=training)
        outputs = self._fc(outputs)
        self.endpoints['head'] = outputs
    return outputs
