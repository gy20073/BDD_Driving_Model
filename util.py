import tensorflow as tf
import numpy as np
import re

def resize_images(images, size, method=0, align_corners=False):
    if '0.10.' == tf.__version__[:5]:
        return tf.image.resize_images(images, size[0], size[1], method, align_corners)
    else:
        return tf.image.resize_images(images, size, method, align_corners)

def to_one_hot_label(sparse_labels_, batch_size, num_classes):
    # Reshape the labels into a dense Tensor of
    # shape [FLAGS.batch_size, num_classes].
    sparse_labels = tf.reshape(sparse_labels_, [batch_size, 1])
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    dense_labels = tf.sparse_to_dense(concated,
                                      [batch_size, num_classes],
                                      1.0, 0.0)
    return dense_labels

def _activation_summary(x, TOWER_NAME):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  print(x)
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def activation_summaries(endpoints, tower_name):
  if isinstance(endpoints, dict):
    end_values = endpoints.values()
  elif isinstance(endpoints, list):
    # throw away the tuple's first entry which is the name
    end_values = [t if isinstance(t, tf.Tensor) else t[1] for t in endpoints]
  else:
    print(endpoints)
    print("unknown endpoint type")

  with tf.name_scope('summaries'):
    print("-"*40 + "\nAll tensors that will be summarized:")
    for act in end_values:
      _activation_summary(act, tower_name)

def bool_select(data, bool):
    # select entries from first dim of data array using bool as indicator
    # only support one dim bool for now
    assert(bool.get_shape().ndims == 1)

    true_loc = tf.where(bool)
    # reshape to 1D
    true_loc = tf.reshape(true_loc, [-1])
    return tf.gather(data, true_loc)


def filter_no_groundtruth_label(label, prediction):
    # treat the negative labels as no ground truth label, filter them out
    assert(label.get_shape().ndims == 1)
    valid = tf.greater_equal(label, 0)
    return bool_select(label, valid), bool_select(prediction, valid)

def bilinearResize(images, ratiox, ratioy):
    '''
    images: 4D image batch
    ratiox, ratioy: magnification ratio. Positive integer.
    '''

    b, h, w, c = [v.value for v in images.get_shape()]
    
    sidex = 2 * ratiox - 1
    sidey = 2 * ratioy - 1

    interpolatex = np.true_divide((ratiox - np.abs(np.arange(sidex) - ratiox + 1)), ratiox)
    interpolatey = np.true_divide((ratioy - np.abs(np.arange(sidey) - ratioy + 1)), ratioy)
    weight = np.outer(interpolatex, interpolatey).astype(np.float32)

    weights = np.zeros((sidex,sidey,c,c), dtype=np.float32)   
    for i in range(c):
        weights[:,:,i,i] = weight

    out_shape = [b, h*ratiox, w*ratioy, c]
    strides = [1, ratiox, ratioy, 1]
    kernel = tf.constant(weights, name='bilinear_convt_weights')

    return tf.nn.conv2d_transpose(images, weights, 
            out_shape, strides=strides, padding='SAME')

def tensors_in_checkpoint_file(file_name):
    try:
        reader = tf.train.NewCheckpointReader(file_name)

        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map.keys()

    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed with SNAPPY.")
        else:
            raise IOError("error loading checkpoint")

def loss_weights(empirical_dist, epsilon):
    # an analogy of all 1 weights, don't change the relative learning rates
    d = np.array(empirical_dist)
    assert len(d.shape) == 1
    assert np.abs(1 - np.sum(d)) < 1e-3

    nbin = d.shape[0]

    # guard against 0
    d = (1 - epsilon) * d + epsilon * (1.0 / nbin)
    # take the inverse
    d = 1.0 / d / nbin

    return d