from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import util
import util_car
from util import activation_summaries
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import init_ops
import math, importlib, sys
import numpy as np
from util import *
import copy, pickle
import scipy.ndimage
from collections import defaultdict
import models.BasicConvLSTMCell as BasicConvLSTMCell
import os
from scipy.ndimage import gaussian_filter

# base CNNs
from models.kaffe.caffenet import CaffeNet
#from models.kaffe.caffenet_dilation import CaffeNet_dilation
from models.kaffe.caffenet_dilation8 import CaffeNet_dilation8

TOWER_NAME = 'tower'
BATCHNORM_MOVING_AVERAGE_DECAY=0.9997
MOVING_AVERAGE_DECAY=0.9999
IGNORE_LABEL = 255

tf.app.flags.DEFINE_string('arch_selection', 'LRCN',
                           """select which arch to use under this file""")
tf.app.flags.DEFINE_string('sub_arch_selection', 'car_stop',
                           """select which sub_arch to use under the arch
                           available archs are car_stop, car_discrete, car_continuous""")


tf.app.flags.DEFINE_string('pretrained_model_path', './data/tf.caffenet.bin',
                           """The pretrained model weights""")

tf.app.flags.DEFINE_string('lstm_hidden_units', "256,256",
                           """define how many hidden layers and the number of hidden units in each of them""")
tf.app.flags.DEFINE_string('conv_lstm_filter_sizes', "3,3",
                           """conv lstm filter sizes""")
tf.app.flags.DEFINE_string('conv_lstm_max_pool_factors', "2,2",
                           """conv lstm max pool factors, 1 for not using max pools""")

tf.app.flags.DEFINE_string('cnn_feature', "fc7",
                           """which layer of CNN feature to use""")
tf.app.flags.DEFINE_integer('projection_dim', 512,
                           """the projection dimesion (using 1*1 conv) before feeding into the LSTM""")
tf.app.flags.DEFINE_string('train_stage_name', "stage_lstm",
                           """which training stage multiplier to use""")

# flags to select different architecture
tf.app.flags.DEFINE_boolean('add_dropout_layer', False,
                           """whether add dropout after image feature layer""")
tf.app.flags.DEFINE_float('keep_prob', 0.1,
                           """the keep probability for dropout layer""")
tf.app.flags.DEFINE_boolean('add_dim_reduction', True,
                           """whether to add dimension reduction layer after last feature layer (after the dropout)""")
tf.app.flags.DEFINE_boolean('no_batch_norm', False,
                           """whether to use batch norm in the whole design""")
tf.app.flags.DEFINE_boolean('weight_decay_exclude_bias', False,
                           """True if do not want the bias to have weight decay""")


tf.app.flags.DEFINE_integer('add_hidden_layer_before_LSTM', -1,
                           """whether to add hidden layer before input to LSTM, if > 0""")
tf.app.flags.DEFINE_boolean('enable_basenet_dropout', False,
                           """Whether to enable the base network's dropout in fc6 and fc7""")
tf.app.flags.DEFINE_float('basenet_keep_prob', -1.0,
                            '''The basenet dropout keep probability, <0 for default''')
tf.app.flags.DEFINE_boolean('add_avepool_after_dim_reduction', False,
                           """whether to add a image level average pooling after the dimension reduction""")
tf.app.flags.DEFINE_integer('add_avepool_after_dim_reduction_with_stride', -1,
                           """add an average pool layer after dim reduction with specified stride, if > 0""")


tf.app.flags.DEFINE_string('image_network_arch', "CaffeNet",
                           """which image base network to use, other options are CaffeNet_dilation8""")
tf.app.flags.DEFINE_string('segmentation_network_arch', "CaffeNet_dilation8",
                           """which image base network to use""")

tf.app.flags.DEFINE_integer('image_preprocess_pad', -1,
                           """How many padding to be added to the image input""")

# for CNN_FC baseline
tf.app.flags.DEFINE_integer('history_window', 9,
                           """in the CNN_FC model how many history frames to look when prediction""")
tf.app.flags.DEFINE_integer('cnn_fc_hidden_units', 64,
                           """in the CNN_FC model, how many hidden units of temporal conv""")
tf.app.flags.DEFINE_boolean('use_image_feature', True,
                           """whether to use CNN feature""")
tf.app.flags.DEFINE_boolean('use_previous_speed_feature', False,
                           """whether to use previous speed""")

# the continuous space discretization method
tf.app.flags.DEFINE_integer('discretize_n_bins', 15,
                            '''how many bins to discretize for course and speed''')
tf.app.flags.DEFINE_float('discretize_max_angle', math.pi / 3,
                            '''max angle of the steering wheel''')
tf.app.flags.DEFINE_float('discretize_min_angle', 0.017,
                            '''min angle of the steering wheel, besides 0''')
tf.app.flags.DEFINE_float('discretize_bound_angle', math.pi,
                            '''the upper bound of the angle, larger than the discretize_max_angle''')

tf.app.flags.DEFINE_float('discretize_max_speed', 30,
                            '''max speed of the car for discretization''')
tf.app.flags.DEFINE_float('discretize_min_speed', 0.3,
                            '''min speed of the car''')
tf.app.flags.DEFINE_float('discretize_bound_speed', 40,
                            '''the upper bound of speed, larger than the discretize_max_speed''')
tf.app.flags.DEFINE_float('discretize_label_gaussian_sigma', 1.0,
                            '''the sigma parameter for gaussian smoothing''')
tf.app.flags.DEFINE_float('discretize_min_prob', 1e-6,
                            '''min prob for each bin, avoid underflow''')
tf.app.flags.DEFINE_string('discretize_bin_type', "log",
                            '''What kind of bins to use for discritization''')
tf.app.flags.DEFINE_string('discretize_datadriven_stat_path', "",
                            '''the file path to the data driven statistics''')

tf.app.flags.DEFINE_integer('city_num_classes', '19',
                            "class number for side train task")
tf.app.flags.DEFINE_integer('side_info_baseline', '0',
                            "0 if we use concatenate features, 1 if we only need image features")
tf.app.flags.DEFINE_integer('PTrain', '0',
                            'if we want to use privileged training set this to 1')
tf.app.flags.DEFINE_float('ptrain_weight', 1.0,
                          'The weight of the privilege loss')
tf.app.flags.DEFINE_boolean('omit_action_loss', False,
                          'Omit the action loss for using the ptrain as pretraining')
tf.app.flags.DEFINE_string('class_balance_path', "",
                            '''Which empirical distribution path to use, if empty then don't use balancing''')
tf.app.flags.DEFINE_float('class_balance_epsilon', 0.01,
                            '''having this prob to draw from a uniform distribution''')

tf.app.flags.DEFINE_string('temporal_net', "LSTM",
                            '''Which temporal net to use, could be LSTM or CNN_FC''')

tf.app.flags.DEFINE_boolean('normalize_before_concat', True,
                            '''normalization before feature concatenation''')
tf.app.flags.DEFINE_string('unique_experiment_name', "",
                            '''the scope name of the architecture''')

tf.app.flags.DEFINE_float('dropout_LSTM_keep_prob', -1,
                            '''dropout probability for LSTM''')
tf.app.flags.DEFINE_boolean('pdf_normalize_bins', True,
                            '''Normalize the pdf for each bin. Disable it for visualization purpose''')
tf.app.flags.DEFINE_string('cnn_split', 'conv4',
                           'where we want to split for priviledged training')
tf.app.flags.DEFINE_boolean('early_split', False, 'whether split the network in an early stage')
tf.app.flags.DEFINE_boolean('image_downsample', False, 'downsample to 384, 216')

tf.app.flags.DEFINE_string('prior_folder_path', '',
                           'the turning / straight prior file path')
tf.app.flags.DEFINE_string('prior_name', '',
                           'filename.prior_name')
tf.app.flags.DEFINE_string('phase', '',
                           'train, eval, stat, rnn_inference')

tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          """L2 regularization.""")

tf.app.flags.DEFINE_string('EWC_Fisher_path', '',
                           'Where the Fisher diagnal matrix is stored')
tf.app.flags.DEFINE_string('EWC_MAP_path', '',
                           'Where the MAP checkpoint is stored')
tf.app.flags.DEFINE_float('EWC_weight', 0.0,
                           'the lambda weight for EWC')

tf.app.flags.DEFINE_boolean('ss_bottleneck_arch', False,
                            'whether to use the bottle neck architecture')
tf.app.flags.DEFINE_integer('image_preprocess_pad_seg', -1,
                           """How many padding to be added to the image input, for segmentation image""")

tf.app.flags.DEFINE_string('action_mapping_arch', "",
                            'which action mapping arch to apply')
tf.app.flags.DEFINE_boolean('action_mapping_loss', False,
                            'whether to supervise the intermediate output with extra data')
tf.app.flags.DEFINE_float('action_mapping_main_weight', 1.0,
                           'weights for the action mapping branch')
# for the hinge loss
tf.app.flags.DEFINE_float('action_mapping_threshold', 0.05,
                           'C * max(thresh, |dist1-dist2|)')
tf.app.flags.DEFINE_float('action_mapping_C', 100.0, '')


FLAGS = tf.app.flags.FLAGS

def convert_name(name, new_prefix):
    if not ("TrainStage" in name):
        return name
    else:
        sp = name.split("/")
        sp[0] = new_prefix
        return "/".join(sp)

def inference(net_inputs, num_classes, for_training=False, scope=None, initial_state=None):
    #weight_decay = 0.0005 # disable weight decay and add all of them back in the end.
    weight_decay = 0.0
    bias_initialize_value = 0.1
    # tunable things end here

    method = globals()[FLAGS.arch_selection]

    if FLAGS.no_batch_norm:
        norm_fn = None
    else:
        norm_fn = slim.batch_norm
    norm_params = {'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
                   'center': True,
                   'scale': True,
                   'epsilon': 0.001,
                   'is_training': for_training}
    # the batch norm's parameter is complete using parameters above

    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training = for_training):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=norm_fn, normalizer_params=norm_params,
                            biases_initializer=tf.constant_initializer(bias_initialize_value),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d], padding='SAME', ):
                # the args left for conv2d: input, num_outputs, kernel_size, stride
                # the args left for fc: input, num_outputs
                with slim.arg_scope([slim.max_pool2d, slim.avg_pool2d], padding='VALID'):
                    # the args left for *_pool2d: kernel_size, stride
                    with tf.op_scope([net_inputs], scope, FLAGS.arch_selection) as sc:
                        end_points_collection = sc + '_end_points'
                        with slim.arg_scope([slim.conv2d, slim.fully_connected,
                                             slim.max_pool2d, slim.avg_pool2d,
                                             slim.batch_norm],
                                            outputs_collections=[end_points_collection]):
                            logits = method(net_inputs, num_classes, for_training, initial_state=initial_state)

    if FLAGS.weight_decay > 1e-8:
        weight_decay = FLAGS.weight_decay
        if FLAGS.weight_decay_exclude_bias:
            decay_set = []
            excluded = []
            included = []
            for v in tf.trainable_variables():
                name = v.op.name
                if "biases" in name:
                    excluded.append(name)
                else:
                    decay_set.append(tf.nn.l2_loss(v))
                    included.append(name)
            print("excluded weight decays are: ", excluded)
            print("included weight decays are: ", included)
        else:
            decay_set = [tf.nn.l2_loss(v) for v in tf.trainable_variables()]

        l2_loss = weight_decay * tf.add_n(decay_set)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2_loss)

    if FLAGS.EWC_weight > 1e-8 and for_training:
        # load Fisher matrix
        fisher = pickle.load(open(FLAGS.EWC_Fisher_path, "rb"))
        # load MAP
        map = pickle.load(open(FLAGS.EWC_MAP_path, "rb"))
        # construct the trainable variable set
        trainable = {}
        for v in tf.trainable_variables():
            trainable[v.op.name] = v

        new_prefix = "TrainStage1_%s" % FLAGS.unique_experiment_name
        ewc_set = []
        for old_name in fisher.keys():
            new_name = convert_name(old_name, new_prefix)
            if new_name not in trainable:
                print ("EWC Warning: %s var not in the new graph, skipping it" % new_name)
                continue
            else:
                print ("EWC: adding %s var with EWC loss" % new_name)
            temp = tf.square(tf.sub(trainable[new_name], map[old_name]))
            temp = tf.mul(temp, fisher[old_name])
            temp = tf.reduce_sum(temp)
            ewc_set.append(temp)
        ewc_loss = tf.mul(FLAGS.EWC_weight, tf.add_n(ewc_set), name="EWC_loss")
        tf.add_to_collection(tf.GraphKeys.LOSSES, ewc_loss)

    # Convert end_points_collection into a end_point dict.
    end_points = tf.get_collection(end_points_collection)
    activation_summaries(end_points, TOWER_NAME)
    return logits

def LRCN(net_inputs, num_classes, for_training, initial_state=None):
    # network inputs are list of tensors:
    # 5D images(NFHWC), valid labels (NF*1), egomotion list (NF*previous_steps*3), this video's name (string)
    if FLAGS.data_provider == "nexar_large_speed":
        images = net_inputs[0]
        speed = net_inputs[1]
        if FLAGS.only_seg == 1:
            assert(FLAGS.city_data == 0)
            seg = net_inputs[3]
            ctx = net_inputs[4]
    else:
        raise ValueError("not a valid dataset")

    ############# the CNN part #############
    # first reshape to 4D, since conv2d only takes in 4D input
    shape = [x.value for x in images.get_shape()]

    if FLAGS.use_image_feature:
        images = tf.cast(images, dtype=tf.float32, name="image_to_float")
        images = tf.reshape(images, [shape[0]*shape[1], shape[2], shape[3], shape[4]])
        if FLAGS.image_downsample:
            images = tf.image.resize_nearest_neighbor(images, [216, 384])
            paddings = tf.constant([[0,0],[6,6],[0,0],[0,0]])
            images = tf.pad(images, paddings, mode='CONSTANT', name='pad_it')
        if FLAGS.low_res:
            paddings = tf.constant([[0,0],[6,6],[0,0],[0,0]])
            images = tf.pad(images, paddings, mode='CONSTANT', name='pad_it')
        

        NET = globals()[FLAGS.image_network_arch]
        if FLAGS.image_preprocess_pad >= 0:
            processed_images = NET.preprocess(images, FLAGS.image_preprocess_pad)
        else:
            processed_images = NET.preprocess(images)
        use_dropout = 0.0
        if FLAGS.enable_basenet_dropout and for_training:
            use_dropout = 1.0
            print("-"*40, "enable basenet dropout")

        if FLAGS.basenet_keep_prob > 0:
            net = NET({"input": processed_images},
                      FLAGS.pretrained_model_path,
                      use_dropout=use_dropout,
                      keep_prob=FLAGS.basenet_keep_prob)
        else:
            net=NET({"input": processed_images},
                    FLAGS.pretrained_model_path,
                    use_dropout=use_dropout)

        image_features = net.layers[FLAGS.cnn_feature]
            
    if FLAGS.unique_experiment_name != "": #TODO: harry add this flag to the setting
        stage_status = "TrainStage1_%s" % FLAGS.unique_experiment_name
    else:
        stage_status = "TrainStage1"
        #note: TrainStage1 vs TrainStage1_Pri for the version changes!
    if FLAGS.city_data:
        city_features = privileged_training(net_inputs, num_classes, for_training, stage_status, images, shape)

    # allow adding the motion tower, even without the city segmentation data.
    image_features = motion_tower(stage_status, image_features)

    if FLAGS.omit_action_loss:
        return [None, city_features]

    print("-"*40, stage_status)
    with tf.variable_scope(stage_status):
        all_features = []

        # the indices that don't wish to be normalized
        normalize_exceptions = []
        if FLAGS.use_image_feature:
            if FLAGS.add_dim_reduction:
                # a projection layer to avoid large feature dims
                image_features = slim.conv2d(image_features, FLAGS.projection_dim, [1, 1], 1, scope="dim_reduction")
            if FLAGS.add_avepool_after_dim_reduction:
                # reduce the H and W dimension
                image_features = tf.reduce_mean(image_features, reduction_indices=[1,2])
            avepool_stride = FLAGS.add_avepool_after_dim_reduction_with_stride
            if FLAGS.add_avepool_after_dim_reduction_with_stride > 0:
                image_features = slim.avg_pool2d(image_features, kernel_size=avepool_stride, stride=avepool_stride)
            if FLAGS.add_dropout_layer:
                image_features = slim.dropout(image_features, keep_prob=FLAGS.keep_prob, scope="dropout_image_features")

            image_feature_dim = [x.value for x in image_features.get_shape()]

            if FLAGS.add_hidden_layer_before_LSTM > 0:
                image_features = tf.reshape(image_features, [shape[0]*shape[1], -1])
                image_features = slim.fully_connected(image_features,
                                                      FLAGS.add_hidden_layer_before_LSTM,
                                                      scope="dim_reduction_before_LSTM")

            # now image features is: batch_size*Frames, H, W, C
            # reshape to batch * Frames * 1 * HWC
            image_features = tf.reshape(image_features, [shape[0], shape[1], -1])
            all_features.append(image_features)

        if FLAGS.use_previous_speed_feature:
            speed = tf.reshape(speed, [shape[0], shape[1], 2])
            shift_amount = 5
            # shift the speed by 5 frames
            speed = tf.pad(speed,
                          [[0, 0], [shift_amount, 0], [0, 0]],
                          mode='CONSTANT',
                          name="pad_speed")
            speed = speed[:, :-shift_amount, :]

            normalize_exceptions.append(len(all_features))
            all_features.append(speed)

        ############# concatenate other information source #############
        if FLAGS.only_seg == 1 :
            ctx_features_tmp = tf.transpose(ctx, perm = [0,1,3,4,2])
            all_features = [tf.reshape(ctx_features_tmp, [shape[0],shape[1], -1])]
        
        if len(all_features) > 1 and FLAGS.normalize_before_concat:
            for i in range(len(all_features)):
                if not (i in normalize_exceptions):
                    all_features[i] = tf.nn.l2_normalize(all_features[i], 2)

        # all_features have shape: B, F, #features
        all_features = tf.concat(2, all_features)
        ############# the RNN temporal part #############
        # get hidden layer config from commandline
        if FLAGS.dropout_LSTM_keep_prob > 0:
            print("-"*40, "dropout before LSTM is applied")
            all_features = slim.dropout(all_features,
                                        keep_prob=FLAGS.dropout_LSTM_keep_prob,
                                        scope="dropout_before_lstm")
        if FLAGS.temporal_net == "TCNN":
            sa=[x.value for x in all_features.get_shape()]
            all_features = tf.reshape(all_features, [sa[0], sa[1], 1, sa[2]])

            nhist = FLAGS.history_window
            all_features = tf.pad(all_features,
                                    [[0, 0], [nhist - 1, 0], [0, 0], [0, 0]],
                                    mode='CONSTANT',
                                    name="pad_ahead")

            # temporal convolution to get the features
            # output is batch * Frames * 1 * cnn_fc_hidden_units
            hidden = slim.conv2d(all_features,
                                 FLAGS.cnn_fc_hidden_units,
                                 [nhist, 1],
                                 stride=1,
                                 padding="VALID",
                                 scope="temporal_conv")

            # reshape to remove batch dimension
            hidden_out = tf.reshape(hidden, [shape[0] * shape[1], -1])
        elif FLAGS.temporal_net == "LSTM":
            ############# the RNN temporal part #############
            # get hidden layer config from commandline

            splits = (FLAGS.lstm_hidden_units).split(",")
            splits = [int(x.strip()) for x in splits]

            # the multilayer stacked LSTM
            lstms = []
            for hidden in splits:
                lstms.append(tf.nn.rnn_cell.BasicLSTMCell(hidden, state_is_tuple=True))
            stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstms, state_is_tuple=True)

            # feed into rnn
            feature_unpacked = tf.unpack(all_features, axis=1)

            if initial_state is not None:
                begin_state = initial_state
            else:
                begin_state = stacked_lstm.zero_state(shape[0], dtype=tf.float32)

            output, state = tf.nn.rnn(stacked_lstm,
                                      feature_unpacked,
                                      dtype=tf.float32,
                                      initial_state=begin_state)
            # TODO: state is not used afterwards

            ################Final Classification#################
            # concatentate outputs into a single tensor, the output size is (batch*nframe, hidden[-1])

            hidden_out = tf.pack(output, axis=1, name='pack_rnn_outputs')
            hidden_out = tf.reshape(hidden_out, [shape[0] * shape[1], -1])

        elif FLAGS.temporal_net.lower() == "convlstm":
            # validate only image features are presented
            # it should have 4 shape coordinates: B*F, H, W, C
            assert(len(image_feature_dim)==4)
            # reshape the all_features into a 5D tensor
            all_features = tf.reshape(all_features, shape[0:2] + image_feature_dim[1:])
            # feed into rnn
            cur_inp = tf.unpack(all_features, axis=1)

            # extract the ConvLSTM architecture parameters
            def str_to_int_list(str):
                splits = str.split(",")
                out = [int(x.strip()) for x in splits]
                return out
            hidden_units = str_to_int_list(FLAGS.lstm_hidden_units)
            filters = str_to_int_list(FLAGS.conv_lstm_filter_sizes)
            pools = str_to_int_list(FLAGS.conv_lstm_max_pool_factors)

            assert(len(hidden_units) == len(filters))
            assert (len(hidden_units) == len(pools))

            for ilayer in range(len(hidden_units)):
                # cur_inp shape B H W C
                input_shape = [x.value for x in cur_inp[0].get_shape()[1:3]]
                conv_lstm = BasicConvLSTMCell.BasicConvLSTMCell(input_shape,
                                                                [filters[ilayer], filters[ilayer]],
                                                                hidden_units[ilayer],
                                                                state_is_tuple=True)

                zero_state = conv_lstm.zero_state(shape[0], dtype=tf.float32)
                cur_inp, state = tf.nn.rnn(conv_lstm,
                                           cur_inp,
                                           dtype=tf.float32,
                                           initial_state=zero_state)
                # TODO: state is not used afterwards

                # the max pool to reduce dimensions
                if pools[ilayer]>1:
                    merged = tf.pack(cur_inp, axis=1, name='concat_before_max_pool')
                    merged_shape = [x.value for x in merged.get_shape()]
                    merged = tf.reshape(merged, [shape[0]*shape[1]]+merged_shape[2:])
                    merged = slim.max_pool2d(merged,
                                             kernel_size=pools[ilayer],
                                             stride=pools[ilayer])
                    merged_shape = [x.value for x in merged.get_shape()]
                    merged = tf.reshape(merged, shape[0:2]+merged_shape[1:])
                    cur_inp = tf.unpack(merged, axis = 1)
            output = cur_inp
            ################Final Classification#################
            # concatentate outputs into a single tensor, the output size is (batch*nframe, H', W', C')
            #hidden_out = tf.concat(0, output, name='concat_rnn_outputs')
            # remove the spatio dimensions to be compatible with the code before
            #hidden_out = tf.reshape(hidden_out, [shape[0]*shape[1], -1])

            hidden_out = tf.pack(output, axis=1, name='pack_rnn_outputs')
            hidden_out = tf.reshape(hidden_out, [shape[0] * shape[1], -1])
        else:
            raise ValueError("temporal_net invalid: %s" % FLAGS.temporal_net)

        if FLAGS.sub_arch_selection == "car_stop":
            scope = "softmax_linear"
            num_classes = 3
        elif FLAGS.sub_arch_selection == "car_discrete":
            num_classes = 6
            scope = "softmax_linear_%s" % (FLAGS.sub_arch_selection)
        elif FLAGS.sub_arch_selection == "car_loc_xy":
            num_classes = FLAGS.discretize_n_bins * 2
            scope = "softmax_linear_%s" % (FLAGS.sub_arch_selection)
            # logit[0] will be the discretized prediction
        elif FLAGS.sub_arch_selection == "car_joint":
            # we discritize the bins jointly into a square number of them
            num_classes = FLAGS.discretize_n_bins ** 2
            scope = "softmax_linear_%s" % (FLAGS.sub_arch_selection)
            # logit[0] will be the discretized prediction
        else:
            raise ValueError("not valid sub_arch_selection")


        logits = [slim.fully_connected(hidden_out,
                                       num_classes,
                                       scope=scope,
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       biases_initializer=tf.zeros_initializer)]
        if FLAGS.city_data:
            logits += [city_features]

    # TODO: call the prior if available.
    if FLAGS.prior_folder_path != '':
        sys.path.append(FLAGS.prior_folder_path)
        sp = FLAGS.prior_name.split(".")
        assert (len(sp) == 2)
        lib = importlib.import_module(sp[0])
        func = getattr(lib, sp[1])

        old_shape = logits[0].get_shape()
        print(old_shape)
        # call the function as a pyfunc
        logits[0] = tf.py_func(func, [logits[0], FLAGS.sub_arch_selection], [tf.float32])[0]
        # set shape
        logits[0].set_shape(old_shape)

    if FLAGS.phase == "rnn_inference":
        logits += [state]

    if FLAGS.action_mapping_arch != "":
        # first make sure that the different functions that modify logits doesn't collide
        assert not (FLAGS.phase == "rnn_inference")
        assert not FLAGS.city_data

        # build the action mapping arch, designed for the joint continuous prediction
        assert FLAGS.sub_arch_selection == "car_joint"

    if FLAGS.action_mapping_arch == "v1":
        hw = FLAGS.discretize_n_bins
        act = tf.reshape(logits[0], [shape[0]*shape[1], hw, hw, 1])
        import models.LocallyConv2d as lc2
        act = lc2.layer_local_conv2d(act, num_output=5, kernel_size=[5, 5], strides=[1, 1], use_relu=True, scope="map1")
        act = lc2.layer_local_conv2d(act, num_output=5, kernel_size=[5, 5], strides=[1, 1], use_relu=True, scope="map2")
        act = lc2.layer_local_conv2d(act, num_output=1, kernel_size=[5, 5], strides=[1, 1], use_relu=False, scope="map3")
        act = tf.reshape(act, [shape[0]*shape[1], hw*hw])

        logits = [act, logits[0]]

    if FLAGS.action_mapping_arch == "v2":
        logits += [slim.fully_connected(hidden_out,
                                       num_classes,
                                       scope=scope+"_nexar",
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       biases_initializer=tf.zeros_initializer)]


    return logits
    # The usage of logits from model.inference() output
    # it's only used as input to the 1. loss function and 2. _classification evaluation

def motion_tower(stage_status, image_features):
    with tf.variable_scope(stage_status):
        with tf.variable_scope('Ptrain'):
            if FLAGS.ss_bottleneck_arch:
                assert (FLAGS.early_split == False)
                image_features = slim.conv2d(image_features, FLAGS.city_num_classes, [1, 1], 1,
                                             normalizer_fn=None,
                                             activation_fn=None,
                                             biases_initializer=init_ops.zeros_initializer,
                                             padding='VALID',
                                             scope='segmentation_fc8',
                                             reuse=True if FLAGS.city_data else False)

                # reduce the dimensionality from the conv5 feature map
                with tf.variable_scope('motion_tower'):
                    # size 224/8 = 28
                    image_features = slim.conv2d(image_features, 64, [5, 5], 3, scope='motion1')
                    # size 9
                    image_features = slim.conv2d(image_features, 128, [5, 5], 3, scope='motion2')
                    # size 3
    return image_features

def privileged_training(net_inputs, num_classes, for_training, stage_status, images, shape):
    if FLAGS.data_provider == "nexar_large_speed":
        city_ims = net_inputs[3]
    else:
        raise ValueError("Bad data provider!")

    city_im_shape = [x.value for x in city_ims.get_shape()]
    city_ims = tf.cast(city_ims, dtype=tf.float32, name="image_to_float")
    city_ims = tf.reshape(city_ims,[city_im_shape[0]*city_im_shape[1], 
                city_im_shape[2], 
                city_im_shape[3],
                city_im_shape[4]])

    NET = globals()[FLAGS.segmentation_network_arch]
    if FLAGS.image_preprocess_pad_seg >= 0:
        processed_images = NET.preprocess(city_ims, FLAGS.image_preprocess_pad_seg)
    else:
        processed_images = NET.preprocess(city_ims)

    use_dropout = 0.0
    if FLAGS.enable_basenet_dropout and for_training:
        use_dropout = 1.0
        print("-"*40, "enable base-net dropout")
    net=NET({"input": processed_images},
            FLAGS.pretrained_model_path,
            use_dropout=use_dropout) 

    if FLAGS.early_split:
        city_features = net.layers[FLAGS.cnn_split]  
    else:
        city_features = net.layers[FLAGS.cnn_feature]

    with tf.variable_scope(stage_status):
        with tf.variable_scope('Ptrain'):
            if FLAGS.early_split:
                #city_features = slim.conv2d(city_features, 256,  [3,3], 1, scope='segmentation_conv5')
                #city_features = slim.conv2d(city_features, 4096, [6,6], 1, rate=2, scope='segmentation_conv6')
                #city_features = slim.dropout(city_features, 0.5, scope='segmentation_drop1')
                # TODO: change 1*1 to larger receptive field.
                city_features = slim.conv2d(city_features, 512, [1,1], 1, scope='segmentation_fc7')
                city_features = slim.dropout(city_features, 0.5, scope='segmentation_drop2')

            city_features = slim.conv2d(city_features, FLAGS.city_num_classes, [1,1], 1,
                                        normalizer_fn=None,
                                        activation_fn=None,
                                        biases_initializer=init_ops.zeros_initializer,
                                        padding='VALID',
                                        scope='segmentation_fc8')

            # city features are the predicted semantic segmentation image
            city_segmentation_features = city_features

            pred = tf.argmax(city_features, 3)
            pred_shape = [x.value for x in pred.get_shape()]

            pred = tf.py_func(segmentation_color, [pred], [tf.uint8])[0]
            pred.set_shape([pred_shape[0], pred_shape[1], pred_shape[2], 3])

            pred = tf.image.resize_nearest_neighbor(pred,[shape[2], shape[3]])
            city_ims = tf.cast(city_ims, tf.uint8)
            pred = tf.concat(2,[city_ims, pred])

            tf.image_summary("segmentation_visualization", pred, max_images=113)

    return city_segmentation_features


def CNN_FC(net_inputs, num_classes, for_training, initial_state=None):
    FLAGS.temporal_net = "TCNN"
    return LRCN(net_inputs, num_classes, for_training)

##################### Various Loss Functions ##########################
# fingerprint: loss(logits, labels)
def loss_car_stop(logits, net_outputs, batch_size=None):
    # net_outputs contains is_stop
    labels = net_outputs[0]    # shape: N * F
    # reshape to 1 dimension
    labels = tf.reshape(labels, [-1])

    prediction = logits[0]      # shape: (N * F) * 2
    # filter the no ground truth data
    labels, prediction = util.filter_no_groundtruth_label(labels, prediction)
    labels_shape = tf.shape(labels)
    effective_batch_size = labels_shape[0]
    num_classes = prediction.get_shape()[-1].value

    dense_labels = util.to_one_hot_label(labels, effective_batch_size, num_classes)

    if FLAGS.class_balance_path!="":
        path = FLAGS.class_balance_path + "_stop.npy"
        empirical_distribution = np.load(path)
        weights = util.loss_weights(empirical_distribution, FLAGS.class_balance_epsilon)
        print("using weighted training: ", weights)
        mask = tf.gather(weights, labels)
    else:
        mask = 1.0
    # Cross entropy loss for the main softmax prediction.
    slim.losses.softmax_cross_entropy(prediction, dense_labels, weight = mask)

def loss_car_discrete(logits, net_outputs, batch_size=None):
    # net_outputs contains is_stop, turn
    dense_labels = net_outputs[1]    # shape: N * F * nclass
    # reshape to 2 dimension
    num_classes = dense_labels.get_shape()[-1].value
    dense_labels = tf.reshape(dense_labels, [-1, num_classes])

    if FLAGS.class_balance_path!="":
        path = FLAGS.class_balance_path + "_discrete.npy"
        empirical_distribution = np.load(path)
        weights = util.loss_weights(empirical_distribution, FLAGS.class_balance_epsilon)
        print("using weighted training: ", weights)
        # assume the label being the max response at that point
        labels = tf.argmax(dense_labels, dimension=1)
        mask = tf.gather(weights, labels)
    else:
        mask = 1.0

    # Cross entropy loss for the main softmax prediction.
    slim.losses.softmax_cross_entropy(logits[0], dense_labels, weight=mask)

##################### Loss Functions of the continous discritized #########
# helper function
def segmentation_color(pred):
    color = {0:[128, 64, 128], 1:[244, 35,232], 2:[ 70, 70, 70],
             3:[102, 102,156], 4:[190,153,153], 5:[153,153,153],
             6:[250, 170, 30], 7:[220,220,  0], 8:[107,142, 35],
             9:[152,251, 152], 10:[70,130,180], 11:[220, 20,60],
             12:[255,  0,  0], 13:[0, 0,  142], 14:[0,  0,  70],
             15:[0, 60,  100], 16:[0, 80, 100], 17:[0,  0, 230],
             18:[119, 11, 32]
             }
    color = defaultdict(lambda: [0,0,0], color)
    shape = pred.shape
    pred = pred.ravel()
    pred = np.asarray([color[i] for i in pred])
    pred = pred.reshape(shape[0],shape[1],shape[2],3)
    return pred.astype(np.uint8)

def get_bins():
    name = "get_bins_%s" % FLAGS.discretize_bin_type
    return globals()[name]()

global datadriven_bins_cache
datadriven_bins_cache = None

def merge_small_bins(bins, minwidth):
    bins = np.squeeze(bins)
    last_id = 0
    output = [bins[0]]
    for i in range(1, len(bins)):
        if bins[i] - bins[last_id] > minwidth:
            output.append(bins[i])
            last_id = i
    return output

def samples_to_bins(samples, nbins, minwidth):
    nquantile = nbins * 100
    bins = np.percentile(samples, np.linspace(0, 100, nquantile))
    merged = merge_small_bins(bins, minwidth)
    len_merged = len(merged)
    out = np.interp(np.linspace(0, len_merged - 1, nbins), range(0, len_merged), merged)
    return out

def get_bins_datadriven():
    global datadriven_bins_cache
    if datadriven_bins_cache is not None:
        return copy.deepcopy(datadriven_bins_cache)

    # default should be 181
    nbins = FLAGS.discretize_n_bins - 1

    # should set this
    path = FLAGS.discretize_datadriven_stat_path
    if not os.path.isfile(path):
        raise ValueError("data driven bins has to provide the stat file")

    dists = np.load(path)
    courses = dists[:, 0]
    courses = courses[np.logical_and(courses > math.radians(-89.9),
                                     courses < math.radians(89.9))]
    speeds = dists[:, 1]
    speeds = speeds[np.logical_and(speeds > 0,
                                   speeds < FLAGS.discretize_max_speed)]

    # TODO: make the constant below into FLAG
    datadriven_bins_cache = (list(samples_to_bins(courses, nbins, np.radians(0.001))),
                             list(samples_to_bins(speeds, nbins, 0)) )

    print(datadriven_bins_cache)
    return datadriven_bins_cache

def get_bins_linear():
    n = FLAGS.discretize_n_bins
    # the minimum must > 0, otherwise is in conflict with the lower bound
    min_s = 0
    max_s = FLAGS.discretize_max_speed
    # this will output n-1 values
    speed_bin = np.arange(min_s, max_s, (max_s-min_s)/(n-1)*0.99999)

    min_c = -FLAGS.discretize_max_angle
    max_c = -min_c
    course_bin = np.arange(min_c, max_c, (max_c-min_c)/(n-2)*0.99999)

    # Note: has to return the bins in list, since will add to them in the latter context
    return list(course_bin), list(speed_bin[1:])

def get_bins_log():
    assert FLAGS.discretize_n_bins % 2 == 1
    def get_logspace(vstart, vend, ntotal):
        # return a log space with ntotal points, the first being vstart, end being vend
        step = math.pow(vend / vstart, 1.0 / (ntotal-1))
        out = [vstart]
        for i in range(ntotal - 1):
            out.append(out[-1] * step)
        return out

    speed_bin = get_logspace(FLAGS.discretize_min_speed,
                             FLAGS.discretize_max_speed,
                             FLAGS.discretize_n_bins - 1)

    ntotal = FLAGS.discretize_n_bins  // 2
    course_bin = get_logspace(-FLAGS.discretize_max_angle,
                             -FLAGS.discretize_min_angle,
                             ntotal) + \
                get_logspace(FLAGS.discretize_min_angle,
                             FLAGS.discretize_max_angle,
                             ntotal)
    return course_bin, speed_bin

def get_bins_custom():
    n=FLAGS.discretize_n_bins
    assert n==22

    course_bin = np.array([0.5, 2, 4, 7, 12, 20, 35, 55, 75, 89])
    '''
    course_bin = np.array([ 0.017,
                            0.028078365755519922,
                            0.046376154323573754,
                            0.07659803667245532,
                            0.12651456999081934,
                            0.41222933,
                            0.69794409,
                            0.98365885,
                            1.2693736 ,
                            1.55508836]) / math.pi * 180
    '''

    course_bin = np.concatenate((-course_bin[::-1], [0], course_bin))
    course_bin = course_bin / 180 * math.pi

    min_s = 0
    max_s = 30.0 * 0.99
    speed_bin = np.arange(min_s, max_s, (max_s - min_s) / (n - 1) * 0.99999)
    speed_bin = speed_bin[1:]

    return list(course_bin), list(speed_bin)

def get_bins_joint():
    course_bin = np.array([0.5, 1.5, 2.5, 4, 6, 9, 13, 18, 24, 31, 39, 48, 58, 69])
    course_bin = np.concatenate((-course_bin[::-1], [0], course_bin))
    course_bin = list(course_bin / 180 * math.pi)

    speed_bin = [0.1, 0.5] + list(range(1, 28))

    # to be compatible with code before, assume we have 29 numbers for each, discritize n bins=30
    assert(len(course_bin) == len(speed_bin))
    assert (len(course_bin) == (FLAGS.discretize_n_bins-1))
    return course_bin, speed_bin

# convert the labels to bins
def course_speed_to_discrete(labels):
    course = labels[:, 0]
    speed = labels[:, 1]

    course_bin, speed_bin= get_bins()

    course = np.digitize(course, course_bin)
    speed = np.digitize(speed, speed_bin)

    return course, speed

def smooth_gaussian(a, sigma, axis=-1):
    # a is an multidimensional array, gaussian filter along axis
    b = scipy.ndimage.filters.gaussian_filter1d(a, sigma, mode='constant', cval=0.0, axis=axis)

    # renormalization of the distribution
    b = b / np.sum(b, axis=axis, keepdims=True)
    return b

def sparse_to_dense_smooth(sparse):

    l = len(sparse)
    out = np.zeros((l, FLAGS.discretize_n_bins), dtype=np.float32)
    for i in range(l):
        out[i, sparse[i]] = 1.0
    out = smooth_gaussian(out, FLAGS.discretize_label_gaussian_sigma, -1)

    return out

def call_label_to_dense_smooth(labels):
    course, speed = course_speed_to_discrete(labels)
    course = sparse_to_dense_smooth(course)
    speed = sparse_to_dense_smooth(speed)

    return [course, speed]

def loss_car_loc_xy(logits, net_outputs, batch_size=None):
    # net_outputs contains is_stop, turn, locs
    future_labels = net_outputs[2]    # shape: N * F * 2
    # reshape to 2 dimension
    num_classes = future_labels.get_shape()[-1].value
    NF = future_labels.get_shape()[0].value * \
         future_labels.get_shape()[1].value
    future_labels = tf.reshape(future_labels, [-1, num_classes])

    dense_course, dense_speed = tf.py_func(call_label_to_dense_smooth,
                              [future_labels],
                              [tf.float32, tf.float32])

    if FLAGS.class_balance_path!="":
        path = FLAGS.class_balance_path + "_continuous.npy"
        dists = np.load(path)

        masks = []
        dense_labels = [dense_course, dense_speed]
        for i in range(2):
            weights = util.loss_weights(dists[i], FLAGS.class_balance_epsilon)
            print("using weighted training: ", weights)
            # assume the label being the max response at that point
            labels = tf.argmax(dense_labels[i], dimension=1)
            mask = tf.gather(weights, labels)
            mask.set_shape([NF])
            masks.append(mask)
    else:
        masks = [1.0, 1.0]

    future_predict = logits[0] # shape: (N*F) * 2Nbins
    n = FLAGS.discretize_n_bins
    slim.losses.softmax_cross_entropy(future_predict[:, 0:n], dense_course,
                                      scope="cross_entropy_loss/course",
                                      weight=masks[0])
    slim.losses.softmax_cross_entropy(future_predict[:, n: ], dense_speed,
                                      scope="cross_entropy_loss/speed",
                                      weight=masks[1])
def city_loss(city_prediction, seg_mask):
    #get shape
    city_pred_shape = [x.value for x in city_prediction.get_shape()]
    city_seg_shape = [x.value for x in seg_mask.get_shape()]

    # reshape & resize seg mask
    seg_mask = tf.reshape(seg_mask,[city_seg_shape[0]*city_seg_shape[1],
                                    city_seg_shape[2],
                                    city_seg_shape[3],
                                    city_seg_shape[4]])
    
    seg_mask = tf.image.resize_nearest_neighbor(seg_mask,
                [city_pred_shape[1],
                city_pred_shape[2]], 
                align_corners=None, 
                name=None)
    
    # reshape and select valid pixels
    city_prediction = tf.reshape(city_prediction, [-1,city_pred_shape[3]])
    seg_mask    = tf.reshape(seg_mask, [-1])

    valid = tf.less(seg_mask, IGNORE_LABEL)
    seg_mask = bool_select(seg_mask, valid)   
    city_prediction = bool_select(city_prediction, valid)
    
    # one hot need dtype=tf,int32, to one hot!
    batch_for_onehot = tf.shape(city_prediction)[0]
    seg_mask        = tf.cast(seg_mask, tf.int32)

    seg_mask        = to_one_hot_label(seg_mask, batch_for_onehot, city_pred_shape[3])
    # compute loss
    seg_loss = slim.losses.softmax_cross_entropy(city_prediction,
                             seg_mask,
                             weight=FLAGS.ptrain_weight)
##################### Loss Functions of the jointly discritized #########
def course_speed_to_joint_bin(labels):
    # each of the labels[i, :] is the course and speed
    # convert each pair to the corresponding bin location
    course, speed = course_speed_to_discrete(labels)

    n = FLAGS.discretize_n_bins
    l = len(course)

    # follow the convention of speed first and speed second
    out = np.zeros((l, n, n), dtype=np.float32)

    for i, item in enumerate(zip(course, speed)):
        ci, si = item
        out[i, ci, si] = 1.0

        # do the gaussian smoothing
        out[i, :, :] = gaussian_filter(out[i, :, :],
                        sigma=FLAGS.discretize_label_gaussian_sigma,
                        mode='constant', cval=0.0)
    # renormalization of the distribution
    out = out / np.sum(out, axis=(1,2), keepdims=True)

    out = np.reshape(out, [l, n*n])
    return out


def loss_car_joint(logits, net_outputs, batch_size=None, masks=None):
    # net_outputs contains is_stop, turn, locs
    future_labels = net_outputs[2]    # shape: N * F * 2
    # reshape to 2 dimension
    num_classes = future_labels.get_shape()[-1].value
    future_labels = tf.reshape(future_labels, [-1, num_classes])

    dense_labels = tf.py_func(course_speed_to_joint_bin,
                              [future_labels],
                              [tf.float32])[0]

    if masks is not None:
        pass
    elif FLAGS.class_balance_path!="":
        path = FLAGS.class_balance_path + "_joint.npy"
        dist = np.load(path)

        weights = util.loss_weights(dist, FLAGS.class_balance_epsilon)
        print("using weighted training: ", weights)
        # assume the label being the max response at that point
        labels = tf.argmax(dense_labels, dimension=1)
        masks = tf.gather(weights, labels)
    else:
        masks = 1.0

    future_predict = logits[0]
    slim.losses.softmax_cross_entropy(future_predict, dense_labels,
                                      weight=masks,
                                      scope="softmax_loss_joint")


def py_is_mkz(names, ntotal):
    ans = []
    assert ntotal % len(names) == 0
    nrepeats = int(ntotal / len(names))
    for name in names:
        try:
            num = int(name)
            if num < 100000:
                ans.append(1.0)
            else:
                ans.append(0.0)
        except:
            ans.append(0.0)

    out = []
    for a in ans:
        for i in range(nrepeats):
            out.append(a)

    return np.array(out, dtype=np.float32)


def loss2joint(logits, net_outputs):
    # compute the weight to see which are nexar / MKZ datasets
    ninstance = logits[0].get_shape()[0].value
    masks = tf.py_func(py_is_mkz,
                       [net_outputs[-1], ninstance],
                       [tf.float32])[0]  # 1.0 if it's a MKZ data TODO: compute which are for MKZ and which are for nexar
    masks.set_shape([ ninstance ])
    # for MKZ
    with tf.variable_scope("MKZ_loss"):
        loss_car_joint([logits[0]], net_outputs, masks=masks * FLAGS.action_mapping_main_weight)
    # for nexar
    with tf.variable_scope("nexar_loss"):
        loss_car_joint([logits[1]], net_outputs, masks=1.0-masks)

    with tf.variable_scope("limit_change_loss"):
        dist = tf.reduce_sum(tf.abs(tf.nn.softmax(logits[0])-tf.nn.softmax(logits[1])), 1)
        losses = FLAGS.action_mapping_C * tf.maximum(0.0, dist - FLAGS.action_mapping_threshold)
        loss = tf.reduce_mean(losses)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

def loss(logits, net_outputs, batch_size=None):
    if FLAGS.city_data:
        #city seg loss
        city_logits = logits[1]
        seg_mask = net_outputs[3]
        city_seg_loss = city_loss(city_logits, seg_mask)
    if FLAGS.omit_action_loss:
        return

    if FLAGS.action_mapping_loss:
        loss2joint(logits, net_outputs)
        return

    func = globals()["loss_%s" % FLAGS.sub_arch_selection]
    func(logits[0:1], net_outputs, batch_size)
    

####################pdf of continous distribution #######################
# only used in continous_pdf_car_loc_xy
def pdf_bins(bins, prob, query):
    # bins has n+1 numbers
    # probs has n probs that sum to 1
    # query is some sample point
    # return the pdf of that point

    # Make sure that bins[0] and bins[-1] are the same for every method
    # otherwise the comparision won't be fair

    if query < bins[0]:
        query = bins[0]
    if query > bins[-1]:
        query = bins[-1]

    assert(len(bins) == len(prob)+1)
    for i in range(len(prob)):
        if bins[i + 1] >= query >= bins[i]:
            if FLAGS.pdf_normalize_bins:
                return prob[i] / (bins[i + 1] - bins[i])
            else:
                return prob[i]


def pdf_bins_batch(bins, prob, querys):
    assert (len(bins) == len(prob) + 1)

    querys = np.array(querys)
    bins = np.array(bins)
    idx = np.digitize(querys, bins[1:-1])

    # get the mass
    masses = prob[idx]

    if FLAGS.pdf_normalize_bins:
        # get the x bin length
        xlen = bins[idx + 1] - bins[idx]
        return masses / xlen
    else:
        return masses



def pdf_bins_batch_2D(cs_bins, prob, querys):
    # assume the input cs_bins are augmented with the bounds
    cbin, sbin = cs_bins
    cbin = np.array(cbin)
    sbin = np.array(sbin)

    assert(len(cbin) == len(sbin))
    prob = np.reshape(prob, (len(cbin)-1, len(cbin)-1))

    querys = np.array(querys)
    cidx = np.digitize(querys[:, 0], cbin[1:-1])
    sidx = np.digitize(querys[:, 1], sbin[1:-1])

    # get the mass
    masses = prob[cidx, sidx]

    if FLAGS.pdf_normalize_bins:
        # get the x bin length
        xlen = cbin[cidx + 1] - cbin[cidx]
        ylen = sbin[sidx + 1] - sbin[sidx]

        return masses / (xlen*ylen)
    else:
        return masses


# this is called from continous_pdf function
def continous_pdf_car_loc_xy(logits, labels):
    # the first entry is the predicted logits
    # first part is course and second part is speed
    logits = logits[0]
    n = FLAGS.discretize_n_bins
    course = logits[:, 0:n]
    speed  = logits[:, n: ]
    # do the softmax
    course = util_car.softmax(course)
    speed  = util_car.softmax(speed)

    # get the bins
    course_bin, speed_bin = get_bins()
    course_bin = [-FLAGS.discretize_bound_angle] + course_bin + \
                 [FLAGS.discretize_bound_angle]
    course_bin = np.array(course_bin)
    speed_bin  = [0] + speed_bin + [FLAGS.discretize_bound_speed]
    speed_bin = np.array(speed_bin)

    # the labels also only has one entry
    labels = labels[0]
    course_label = labels[:, 0]
    speed_label  = labels[:, 1]
    out = np.zeros((labels.shape[0],2), dtype=np.float32)

    for i in range(labels.shape[0]):
        out[i, 0] = pdf_bins(course_bin, course[i,:], course_label[i])
        out[i, 1] = pdf_bins(speed_bin, speed[i, :], speed_label[i])

    out = np.log(out + FLAGS.discretize_min_prob)

    # return the log prob of each speed and course
    return out

def continous_pdf_car_joint(logits, labels):
    logits = logits[0]
    softmaxed = util_car.softmax(logits)

    # get the bins
    course_bin, speed_bin = get_bins()
    course_bin = [-FLAGS.discretize_bound_angle] + course_bin + \
                 [FLAGS.discretize_bound_angle]
    speed_bin = [0] + speed_bin + [FLAGS.discretize_bound_speed]

    out = []
    labels = labels[0]
    for i in range(labels.shape[0]):
        this = pdf_bins_batch_2D([course_bin, speed_bin], softmaxed[i, :], labels[i:(i+1),:])
        out.append(this)
    out = np.array(out)
    out = np.log(out + FLAGS.discretize_min_prob)

    # return the log prob, Note that this output is not compatible any more
    return out


# used in the wrapper and draw_sector
def multi_querys_car_loc_xy_impl(logits, querys):
    # the first entry is the predicted logits
    # first part is course and second part is speed
    logits = logits[0]
    n = FLAGS.discretize_n_bins
    course = logits[:, 0:n]
    speed = logits[:, n:]
    # do the softmax
    course = util_car.softmax(course)
    speed = util_car.softmax(speed)

    # get the bins
    course_bin, speed_bin = get_bins()
    course_bin = [-FLAGS.discretize_bound_angle] + course_bin + \
                 [FLAGS.discretize_bound_angle]
    speed_bin = [0] + speed_bin + [FLAGS.discretize_bound_speed]

    # the labels also only has one entry
    course_querys = copy.deepcopy(querys[0])
    speed_querys = copy.deepcopy(querys[1])

    course_querys = pdf_bins_batch(course_bin, course[0, :], course_querys)
    speed_querys = pdf_bins_batch(speed_bin, speed[0, :], speed_querys)

    # return the log prob of each speed and course
    return [course_querys, speed_querys]


def multi_querys_car_loc_xy(logits, querys):
    querys = np.array(querys)
    course_querys, speed_querys = \
        multi_querys_car_loc_xy_impl(logits, [querys[:, 0], querys[:, 1]])
    return course_querys * speed_querys


# change the definition of this function to input a list of (c, s) pairs
def multi_querys_car_joint(logits, querys):
    logits = logits[0]
    softmaxed = util_car.softmax(logits)

    # get the bins
    course_bin, speed_bin = get_bins()
    course_bin = [-FLAGS.discretize_bound_angle] + course_bin + \
                 [FLAGS.discretize_bound_angle]
    speed_bin = [0] + speed_bin + [FLAGS.discretize_bound_speed]

    out = pdf_bins_batch_2D([course_bin, speed_bin], softmaxed[0, :], querys)

    return out


def continous_pdf(logits, labels, prefix="continous_pdf"):
    # logits are the list of logit outputed by the network
    # labels are the list of ground truth labels
    # both of them are python object, not tensorflow object
    func = globals()["%s_%s" % (prefix, FLAGS.sub_arch_selection)]
    return func(logits, labels)
    # return the log prob of the observations


########## MAP of continous distribution #################
def continous_MAP_car_loc_xy_log(logits):
    logits = logits[0]
    n = FLAGS.discretize_n_bins
    course = logits[:, 0:n]
    speed  = logits[:, n: ]

    # return the interpolated values, course and speed could be distribution
    course_bin, speed_bin = get_bins()

    c_factor = course_bin[0] / course_bin[1]
    course_bin = np.array(course_bin)
    course_bin = np.append(course_bin, course_bin[-1] * c_factor)

    s_factor = speed_bin[1] / speed_bin[0]
    speed_bin = np.array(speed_bin)
    speed_bin = np.append(speed_bin, speed_bin[-1] * s_factor)

    course = np.argmax(course, axis=1)
    course_inter = course_bin[course]
    for i in range(len(course)):
        # 0-6 7 8-14, n=15, n//2=7
        if course[i] < FLAGS.discretize_n_bins // 2:
            course_inter[i] *= math.sqrt(c_factor)
        elif course[i] > FLAGS.discretize_n_bins // 2:
            course_inter[i] /= math.sqrt(c_factor)
        else:
            course_inter[i] = 0

    speed = np.argmax(speed, axis=1)
    speed_inter = speed_bin[speed] / math.sqrt(s_factor)

    return np.stack((course_inter, speed_inter), axis=1)

def continous_MAP_car_loc_xy_linear(logits):
    logits = logits[0]
    n = FLAGS.discretize_n_bins
    predicts = [logits[:, 0:n], logits[:, n: ]]

    # return the interpolated values, course and speed could be distribution
    cs_bins = get_bins()

    inters=[]
    # for in range and speed
    for i in range(2):
        bins = cs_bins[i]
        step = bins[1] - bins[0]
        bins.append(bins[-1]+step)
        bins = np.array(bins)

        predicted = np.argmax(predicts[i], axis=1)
        inter = bins[predicted] - step/2
        inters.append(inter)
    return np.stack(inters, axis=1)

def MAP_custom_bin_helper(predicted_cs_argmax):
    # return the interpolated values, course and speed could be distribution
    cs_bins = get_bins()

    inters = []
    # for in range and speed
    appends = [89.9 / 180 * math.pi, 29.99]
    prepend = [-89.9 / 180 * math.pi, 0.001]
    for i in range(2):
        predicted = predicted_cs_argmax[i]

        bins = cs_bins[i]
        bins.append(appends[i])
        bins = np.array(bins)

        bins = np.insert(bins, 0, prepend[i])
        inter = (bins[predicted + 1] + bins[predicted]) / 2.0
        inters.append(inter)
    return inters

def continous_MAP_car_loc_xy_custom(logits):
    logits = logits[0]
    n = FLAGS.discretize_n_bins
    predicts = [logits[:, 0:n], logits[:, n: ]]

    predicted_cs_argmax = [np.argmax(predicts[0], axis=1), np.argmax(predicts[1], axis=1)]
    inters = MAP_custom_bin_helper(predicted_cs_argmax)

    small_angle = 0.3/180*math.pi
    inters[0][-small_angle < inters[0] < small_angle] = 0.0

    return np.stack(inters, axis=1) # will return a #samples * 2 array

def continous_MAP_car_loc_xy_datadriven(logits):
    return continous_MAP_car_loc_xy_custom(logits)

def continous_MAP_car_joint_joint(logits):
    logits = logits[0]
    n = int(FLAGS.discretize_n_bins)

    argm = np.argmax(logits, axis=1)
    predicted_cs_argmax = [(argm / n).astype(int), (argm % n).astype(int)]
    inters = MAP_custom_bin_helper(predicted_cs_argmax)

    return np.stack(inters, axis=1)  # will return a #samples * 2 array


# We should pick the largest density!!!
# all of our implementation here is largest bin
def continous_MAP(logits, return_second_best=False):
    func = globals()["continous_MAP_%s_%s" %
                     (FLAGS.sub_arch_selection, FLAGS.discretize_bin_type)]
    if return_second_best and FLAGS.discretize_bin_type!="joint":
        logits = copy.deepcopy(logits)
        logits = [np.array(logits[0])]
        n = int(FLAGS.discretize_n_bins)
        half = int(n / 2)

        if n % 2 == 0:
            lb = half - 1
            ub = half + 1
            logits[0][:, lb:ub] = -99999
        else:
            # using n/2 will have a bug
            logits[0][:, half] = -99999
    return func(logits)

########################Custom learning rates##########################

# multipliers for the uninitialized part
def stage_lstm():
    stage_name = "TrainStage1"
    ans = {}
    for v in tf.all_variables():
        if stage_name in v.op.name:
            ans[v.op.name] = 1.0
        else:
            ans[v.op.name] = 0.0
    return ans

def stage_all():
    # use the same weights everywhere
    return {}

def stage_classic_finetune():
    stage_name = "TrainStage1"
    ans = {}
    for v in tf.all_variables():
        if stage_name in v.op.name:
            ans[v.op.name] = 10.0
        else:
            ans[v.op.name] = 1.0
    return ans

def learning_rate_multipliers():
    method = globals()[FLAGS.train_stage_name]
    return method()
