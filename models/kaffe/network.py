import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.python.ops import nn
import math


DEFAULT_PADDING = 'SAME'


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    layer_decorated._original=op
    return layer_decorated


class Network(object):

    def __init__(self, inputs, net_weights, trainable=True, use_dropout=0.0):
        print("using Yang's load pretrained weights version")
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = use_dropout
        # net_weights initializer, could be None if don't want to restore weights
        if net_weights:
            self.data_dict = np.load(net_weights).item()
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    # not used, since using Yang's version.
    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, basestring):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def get_saved_value(self, name):
        assert(hasattr(self, 'data_dict'))
        scp = tf.get_variable_scope()
        # assume its name's last index is this layer's name
        scp_name = scp.name.split("/")[-1]

        # print(self.data_dict[scp_name][name])
        '''Creates a new TensorFlow variable.'''
        print(scp_name, name, self.data_dict[scp_name][name].shape)
        return self.data_dict[scp_name][name]

    def make_var(self, name, shape):
        if hasattr(self, 'data_dict'):
            try:
                # assume it's a new variable
                v = tf.get_variable(name,
                                   trainable=self.trainable,
                                   initializer=np.reshape(self.get_saved_value(name), shape) )
            except ValueError:
                # it has existed
                scope = tf.get_variable_scope()
                scope.reuse_variables()
                print("make scope ", scope, " reuse=True")
                v = tf.get_variable(name)
            return v

        else:
            # TODO: Caveat: the initializer for vars with diff shapes are different
            if len(shape) > 1:
                return tf.get_variable(name, shape,
                                       trainable=self.trainable,
                                       initializer=initializers.xavier_initializer())
            else:
                return tf.get_variable(name, shape,
                                       trainable=self.trainable,
                                       initializer=tf.constant_initializer(0.0))


    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def fn_defined(self, input, fn, name):
        return fn(input)

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True,
             rate=1,
             weight_decay=0.0):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        if rate == 1:
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        else:
            convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, padding=padding, rate=rate)

        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            print("weight decay inside network.py = ", weight_decay)
            if weight_decay > 1e-8:
                reg = weight_decay * tf.nn.l2_loss(kernel, name+'_weight_norm')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            print(output.get_shape())
            return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc_bak(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def fc(self, input, num_out, name, relu=True, preload=True, conv_padding="VALID"):
        if preload:
            input_shape = input.get_shape()
            assert (input_shape.ndims == 4)
            input_channel = input_shape[-1].value
            # calculate the k_h and k_w
            nele = (self.data_dict[name]["weights"]).size
            # assume k_h == k_w
            hw = int(math.sqrt(nele*1.0 / num_out / input_channel))
            assert(hw*hw*num_out*input_channel == nele)

            conv_origin=self.conv._original

            return conv_origin(self, input,
                  hw, hw, num_out,
                  1, 1,
                  name,
                  relu=relu,
                  padding=conv_padding,
                  group=1,
                  biased=True)
        else:
            return self.fc_bak(input, num_out, name, relu=relu)

    @staticmethod
    def preprocess(images, mean_bgr = [104.0069, 116.6687, 122.6789]):
        '''
        pre-process the images tensor to fit this network's requirement
        Args:
           images: an input tensor that is at least 4 dimensional (NHWC),
           but could also be higher dimensional, such as NFHWC for videos.
           But the last dimension should always be channel.

           images are also expected to be range from 0 to 255. Both int and float
           types are accepted.
        Returns:
           an op that pre-process the images to fit the network
           will return in dtype=float32

        '''
        shape = images.get_shape()
        assert (shape.ndims >= 4)
        assert (shape[-1].value == 3)

        # the mean
        mean_BGR = np.array(mean_bgr, dtype=np.float32)
        # the mean shape that fits the input images
        mean_shape = tuple([1 for _ in range(shape.ndims - 1)] + [3])
        mean_BGR = np.reshape(mean_BGR, mean_shape)

        # channel swap
        reverse_dim = [False for _ in range(shape.ndims - 1)] + [True]
        images = tf.reverse(images, reverse_dim)

        # cast type
        images = tf.cast(images, tf.float32)

        # mean subtraction
        images = images - mean_BGR

        return images

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name)

    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False, is_training=False):
        with tf.variable_scope(name) as scope:
            norm_params = {'decay':0.999, 'scale':scale_offset, 'epsilon':0.001, 'is_training':is_training,
                           'activation_fn':tf.nn.relu if relu else None}            
            if hasattr(self, 'data_dict'):
                param_inits={'moving_mean':self.get_saved_value('mean'),
                             'moving_variance':self.get_saved_value('variance')}
                if scale_offset:
                    param_inits['beta']=self.get_saved_value('offset')
                    param_inits['gamma']=self.get_saved_value('scale')

                shape = [input.get_shape()[-1]]
                for key in param_inits:
                    param_inits[key] = np.reshape(param_inits[key], shape)
                norm_params['param_initializers'] = param_inits
            # TODO: there might be a bug if reusing is enabled.
            return slim.batch_norm(input, **norm_params)

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)
