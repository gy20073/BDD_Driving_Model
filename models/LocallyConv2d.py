# locally convolution layer adapted from keras, since tensorflow 0.11 don't have keras integrated yet

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf

def int_shape(x):
  """Returns the shape tensor or variable as a tuple of int or None entries.
  Arguments:
      x: Tensor or variable.
  Returns:
      A tuple of integers (or None entries).
  Examples:
  ```python
      >>> from keras import backend as K
      >>> input = K.placeholder(shape=(2, 4, 5))
      >>> K.int_shape(input)
      (2, 4, 5)
      >>> val = np.array([[1, 2], [3, 4]])
      >>> kvar = K.variable(value=val)
      >>> K.int_shape(kvar)
      (2, 2)
  ```
  """
  try:
    return tuple(x.get_shape().as_list())
  except ValueError:
    return None

def ndim(x):
  """Returns the number of axes in a tensor, as an integer.
  Arguments:
      x: Tensor or variable.
  Returns:
      Integer (scalar), number of axes.
  Examples:
  ```python
      >>> from keras import backend as K
      >>> input = K.placeholder(shape=(2, 4, 5))
      >>> val = np.array([[1, 2], [3, 4]])
      >>> kvar = K.variable(value=val)
      >>> K.ndim(input)
      3
      >>> K.ndim(kvar)
      2
  ```
  """
  dims = x.get_shape()._dims
  if dims is not None:
    return len(dims)
  return None

def reshape(x, shape):
  """Reshapes a tensor to the specified shape.
  Arguments:
      x: Tensor or variable.
      shape: Target shape tuple.
  Returns:
      A tensor.
  """
  return array_ops.reshape(x, shape)


def new_concat(values, axis, name="concat"):
    return tf.concat(axis, values, name=name)

def batch_dot(x, y, axes=None):
  """Batchwise dot product.
  `batch_dot` is used to compute dot product of `x` and `y` when
  `x` and `y` are data in batch, i.e. in a shape of
  `(batch_size, :)`.
  `batch_dot` results in a tensor or variable with less dimensions
  than the input. If the number of dimensions is reduced to 1,
  we use `expand_dims` to make sure that ndim is at least 2.
  Arguments:
      x: Keras tensor or variable with `ndim >= 2`.
      y: Keras tensor or variable with `ndim >= 2`.
      axes: list of (or single) int with target dimensions.
          The lengths of `axes[0]` and `axes[1]` should be the same.
  Returns:
      A tensor with shape equal to the concatenation of `x`'s shape
      (less the dimension that was summed over) and `y`'s shape
      (less the batch dimension and the dimension that was summed over).
      If the final rank is 1, we reshape it to `(batch_size, 1)`.
  Examples:
      Assume `x = [[1, 2], [3, 4]]` and `y = [[5, 6], [7, 8]]`
      `batch_dot(x, y, axes=1) = [[17, 53]]` which is the main diagonal
      of `x.dot(y.T)`, although we never have to calculate the off-diagonal
      elements.
      Shape inference:
      Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
      If `axes` is (1, 2), to find the output shape of resultant tensor,
          loop through each dimension in `x`'s shape and `y`'s shape:
      * `x.shape[0]` : 100 : append to output shape
      * `x.shape[1]` : 20 : do not append to output shape,
          dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
      * `y.shape[0]` : 100 : do not append to output shape,
          always ignore first dimension of `y`
      * `y.shape[1]` : 30 : append to output shape
      * `y.shape[2]` : 20 : do not append to output shape,
          dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
      `output_shape` = `(100, 30)`
  ```python
      >>> x_batch = K.ones(shape=(32, 20, 1))
      >>> y_batch = K.ones(shape=(32, 30, 20))
      >>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
      >>> K.int_shape(xy_batch_dot)
      (32, 1, 30)
  ```
  """
  if isinstance(axes, int):
    axes = (axes, axes)
  x_ndim = ndim(x)
  y_ndim = ndim(y)
  if x_ndim > y_ndim:
    diff = x_ndim - y_ndim
    y = array_ops.reshape(y,
                          new_concat(
                              [array_ops.shape(y), [1] * (diff)], axis=0))
  elif y_ndim > x_ndim:
    diff = y_ndim - x_ndim
    x = array_ops.reshape(x,
                          new_concat(
                              [array_ops.shape(x), [1] * (diff)], axis=0))
  else:
    diff = 0
  if ndim(x) == 2 and ndim(y) == 2:
    if axes[0] == axes[1]:
      out = math_ops.reduce_sum(math_ops.multiply(x, y), axes[0])
    else:
      out = math_ops.reduce_sum(
          math_ops.multiply(array_ops.transpose(x, [1, 0]), y), axes[1])
  else:
    if axes is not None:
      adj_x = None if axes[0] == ndim(x) - 1 else True
      adj_y = True if axes[1] == ndim(y) - 1 else None
    else:
      adj_x = None
      adj_y = None

    out = math_ops.batch_matmul(x, y, adj_x=adj_x, adj_y=adj_y)
  if diff:
    if x_ndim > y_ndim:
      idx = x_ndim + y_ndim - 3
    else:
      idx = x_ndim - 1
    out = array_ops.squeeze(out, list(range(idx, idx + diff)))
  if ndim(out) == 1:
    out = tf.expand_dims(out, 1)
  return out


def local_conv2d(inputs,
                 kernel,
                 kernel_size,
                 strides,
                 output_shape,
                 data_format="channels_last"):
  """Apply 2D conv with un-shared weights.
  Arguments:
      inputs: 4D tensor with shape:
              (batch_size, filters, new_rows, new_cols)
              if data_format='channels_first'
              or 4D tensor with shape:
              (batch_size, new_rows, new_cols, filters)
              if data_format='channels_last'.
      kernel: the unshared weight for convolution,
              with shape (output_items, feature_dim, filters)
      kernel_size: a tuple of 2 integers, specifying the
                   width and height of the 2D convolution window.
      strides: a tuple of 2 integers, specifying the strides
               of the convolution along the width and height.
      output_shape: a tuple with (output_row, output_col)
      data_format: the data format, channels_first or channels_last
  Returns:
      A 4d tensor with shape:
      (batch_size, filters, new_rows, new_cols)
      if data_format='channels_first'
      or 4D tensor with shape:
      (batch_size, new_rows, new_cols, filters)
      if data_format='channels_last'.
  Raises:
      ValueError: if `data_format` is neither
                  `channels_last` or `channels_first`.
  """
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format ' + str(data_format))

  stride_row, stride_col = strides
  output_row, output_col = output_shape
  kernel_shape = int_shape(kernel)
  feature_dim = kernel_shape[1]
  filters = kernel_shape[2]

  xs = []
  for i in range(output_row):
    for j in range(output_col):
      slice_row = slice(i * stride_row, i * stride_row + kernel_size[0])
      slice_col = slice(j * stride_col, j * stride_col + kernel_size[1])
      if data_format == 'channels_first':
        xs.append(
            reshape(inputs[:, :, slice_row, slice_col], (1, -1, feature_dim)))
      else:
        xs.append(
            reshape(inputs[:, slice_row, slice_col, :], (1, -1, feature_dim)))

  x_aggregate = new_concat(xs, axis=0)
  output = batch_dot(x_aggregate, kernel)
  output = reshape(output, (output_row, output_col, -1, filters))

  if data_format == 'channels_first':
    output = tf.transpose(output, (2, 3, 0, 1))
  else:
    output = tf.transpose(output, (2, 0, 1, 3))
  return output


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
  """Determines output length of a convolution given input length.
  Arguments:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.
      dilation: dilation rate, integer.
  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  assert padding in {'same', 'valid', 'full'}
  dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
  if padding == 'same':
    output_length = input_length
  elif padding == 'valid':
    output_length = input_length - dilated_filter_size + 1
  elif padding == 'full':
    output_length = input_length + dilated_filter_size - 1
  return (output_length + stride - 1) // stride


def bias_add(x, bias, data_format=None):
  """Adds a bias vector to a tensor.
  Arguments:
      x: Tensor or variable.
      bias: Bias tensor to add.
      data_format: string, `"channels_last"` or `"channels_first"`.
  Returns:
      Output tensor.
  Raises:
      ValueError: In one of the two cases below:
                  1. invalid `data_format` argument.
                  2. invalid bias shape.
                     the bias should be either a vector or
                     a tensor with ndim(x) - 1 dimension
  """
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format ' + str(data_format))
  bias_shape = int_shape(bias)
  if len(bias_shape) != 1 and len(bias_shape) != ndim(x) - 1:
    raise ValueError(
        'Unexpected bias dimensions %d, expect to be 1 or %d dimensions' %
        (len(bias_shape), ndim(x)))
  if ndim(x) == 5:
    if data_format == 'channels_first':
      if len(bias_shape) == 1:
        x += reshape(bias, (1, bias_shape[0], 1, 1, 1))
      else:
        x += reshape(bias, (1, bias_shape[3]) + bias_shape[:3])
    elif data_format == 'channels_last':
      if len(bias_shape) == 1:
        x += reshape(bias, (1, 1, 1, bias_shape[0]))
      else:
        x += reshape(bias, (1,) + bias_shape)
  elif ndim(x) == 4:
    if data_format == 'channels_first':
      if len(bias_shape) == 1:
        x += reshape(bias, (1, bias_shape[0], 1, 1))
      else:
        x += reshape(bias, (1, bias_shape[2]) + bias_shape[:2])
    elif data_format == 'channels_last':
      if len(bias_shape) == 1:
        x = tf.nn.bias_add(x, bias, data_format='NHWC')
      else:
        x += reshape(bias, (1,) + bias_shape)
  elif ndim(x) == 3:
    if data_format == 'channels_first':
      if len(bias_shape) == 1:
        x += reshape(bias, (1, bias_shape[0], 1))
      else:
        x += reshape(bias, (1, bias_shape[1], bias_shape[0]))
    elif data_format == 'channels_last':
      if len(bias_shape) == 1:
        x += reshape(bias, (1, 1, bias_shape[0]))
      else:
        x += reshape(bias, (1,) + bias_shape)
  else:
    x = tf.nn.bias_add(x, bias)
  return x


def add_weight(shape, initializer, name, regularizer=None, constraint=None):
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=tf.float32,
                           trainable=True)

# need to set use_relu, scope parameters
def layer_local_conv2d(inputs, num_output, kernel_size, strides, padding="same",
                       data_format="channels_last", use_relu=True, use_bias=True,  scope=None):
    if padding == "same":
        assert kernel_size[0] % 2 == 1
        assert kernel_size[1] % 2 == 1
        pad0 = kernel_size[0] / 2
        pad1 = kernel_size[1] / 2

        with tf.variable_scope(scope):
            inputs = tf.pad(inputs,
                            [[0, 0], [pad0, pad0], [pad1, pad1], [0, 0]],
                            mode="CONSTANT",
                            name="same_padding")
    elif padding == "valid":
        pass
    else:
        raise ValueError("wrong padding param")
    padding = "valid"

    # input should be NHWC
    input_shape = int_shape(inputs)
    if data_format == 'channels_last':
        input_row, input_col = input_shape[1:-1]
        input_filter = input_shape[3]
    else:
        input_row, input_col = input_shape[2:]
        input_filter = input_shape[1]
    if input_row is None or input_col is None:
        raise ValueError('The spatial dimensions of the inputs to '
                         ' a LocallyConnected2D layer '
                         'should be fully-defined, but layer received '
                         'the inputs shape ' + str(input_shape))

    filters = num_output

    output_row = conv_output_length(input_row, kernel_size[0], padding, strides[0])
    output_col = conv_output_length(input_col, kernel_size[1], padding, strides[1])

    kernel_shape = (output_row * output_col, kernel_size[0] * kernel_size[1] * input_filter, filters)


    with tf.variable_scope(scope):
        kernel=add_weight(shape=kernel_shape, initializer=None, name='kernel')
        if use_bias:
            bias = add_weight(shape=(output_row, output_col, filters), initializer=None, name='bias')

    output = local_conv2d(inputs,
                          kernel,
                          kernel_size,
                          strides,
                          (output_row, output_col),
                          data_format)
    if use_bias:
        output = bias_add(output, bias, data_format=data_format)

    if use_relu:
        output = tf.nn.relu(output)

    return output
