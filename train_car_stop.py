import math
import sys
import os
from subprocess import call
import inspect
sys.path.append('../')

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

# do not change these config once run a first experiment to make the experiments reproducable

def common_config(phase):
    # not doing the script copying yet
    if phase == "train":
        FLAGS.subset = "train"
    elif phase == "eval":
        FLAGS.subset = "validation"
    elif phase == "board":
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "python"

    # related to training
    FLAGS.batch_size = 1
    FLAGS.log_device_placement = False
    FLAGS.data_provider = "nexar_dataset"
    FLAGS.optimizer = "sgd"
    FLAGS.data_dir = "/home/yang/si/data/nexar_100"
    FLAGS.profile = False
    FLAGS.model_definition = "car_stop_model"
    FLAGS.num_readers = 2
    FLAGS.pretrained_model_checkpoint_path = ""
    FLAGS.num_preprocess_threads = 4
    FLAGS.display_loss = 10
    FLAGS.display_summary = 100
    FLAGS.checkpoint_interval = 1000
    FLAGS.input_queue_memory_factor = 8
    FLAGS.momentum = 0.9
    FLAGS.examples_per_shard=1
    FLAGS.use_MIMO_inputs_pipeline=True

    # related to evaluation
    FLAGS.run_once = True
    FLAGS.eval_interval_secs=600
    FLAGS.subsample_factor=1
    FLAGS.eval_method="car_stop"

    # model related
    FLAGS.pretrained_model_path = '/home/yang/si/data/pretrained_models/tf.caffenet.bin'

def common_discrete_small_settings(phase, tag, isFCN, visEval):
    if phase == "train":
        FLAGS.n_sub_frame = 45 if isFCN else 108
    elif phase=="eval":
        FLAGS.balance_drop_prob = -1.0
        FLAGS.n_sub_frame = 100

        FLAGS.eval_method = "car_discrete"

        if visEval:
            FLAGS.output_visualizations=True
            FLAGS.subsample_factor=10
            FLAGS.run_once = True
        else:
            FLAGS.output_visualizations = False
            FLAGS.eval_interval_secs = 10
            FLAGS.run_once = False
    elif phase == "stat":
        set_gpu("0")
        FLAGS.subset="train"
        FLAGS.n_sub_frame = 108

        FLAGS.stat_output_path = "data/" + tag + "/empirical_dist"
        FLAGS.eval_method = "stat_labels"
        FLAGS.no_image_input = True
        FLAGS.subsample_factor = 10

    FLAGS.batch_size = 1 * FLAGS.num_gpus

    # the parameter setting for this experiment
    FLAGS.sub_arch_selection = "car_discrete"
    FLAGS.lstm_hidden_units = "64"
    # dim reduction
    FLAGS.add_dim_reduction = True
    FLAGS.projection_dim = 64
    # dropout
    FLAGS.add_dropout_layer = False

    # some data provider
    FLAGS.decode_downsample_factor = 1
    FLAGS.temporal_downsample_factor = 5

    FLAGS.data_provider = "nexar_large_speed"
    # some tunable ground truth maker
    FLAGS.speed_limit_as_stop = 2.0  # TODO: make sure it make sense
    FLAGS.stop_future_frames = 1 # make sure this make sense
    FLAGS.deceleration_thres = 1
    FLAGS.no_slight_turn = True

    # change to dilation
    FLAGS.image_network_arch = "CaffeNet_dilation" if isFCN else "CaffeNet"
    FLAGS.pretrained_model_path = "/data/yang/si/data/pretrained_models/tf.caffenet.bin"
    FLAGS.cnn_feature = "fc7"

    # learning rates
    FLAGS.num_epochs_per_decay = 4 if isFCN else 2
    FLAGS.initial_learning_rate = 1e-4
    FLAGS.learning_rate_decay_factor = 0.5

    FLAGS.train_stage_name = 'stage_all'

    FLAGS.clip_gradient_threshold = 10.0
    FLAGS.momentum = 0.99

    FLAGS.num_batch_join = 1 if phase == "eval" else 4
def common_lowres_settings(phase):
    if phase == 'train':
        FLAGS.num_readers = 4
        FLAGS.num_preprocess_threads = 8
    FLAGS.IM_WIDTH = 384
    FLAGS.IM_HEIGHT = 216
    FLAGS.temporal_downsample_factor = 1
    FLAGS.n_sub_frame = 110
    FLAGS.stop_future_frames = 1 # change to 1?
    FLAGS.FRAMES_IN_SEG = 110
    FLAGS.frame_rate = 1
    FLAGS.low_res = True
    FLAGS.batch_size = 3 * FLAGS.num_gpus
    


def common_config_post(phase):
    FLAGS.eval_dir = os.path.join(FLAGS.train_dir, "eval")
    FLAGS.checkpoint_dir = FLAGS.train_dir

def flags_to_cmd():
    # dict of flags to values
    d = FLAGS.__dict__["__flags"]
    out=[]
    for k, v in d.iteritems():
        print(k, v)
        out.append("--"+k+"="+str(v))
    return out

def train():
    call(["python", "train.py"] + flags_to_cmd())

def eval():
    call(["python", "eval.py"] + flags_to_cmd())

def test():
    call(["python", "eval.py"] + flags_to_cmd())

def tensorboard():
    call(["tensorboard",
          "--port="+str(FLAGS.tensorboard_port),
          "--logdir="+str(FLAGS.train_dir)])

def stat():
    call(["python", "gather_stat.py"] + flags_to_cmd())

def set_gpu(gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    num_gpus = len(gpus.split(","))
    FLAGS.num_gpus = num_gpus

# the two functions below will be called by small_config
def tunable_config():
    # related to the model
    FLAGS.arch_selection = "LRCN"
    FLAGS.lstm_hidden_units = "256"
    FLAGS.cnn_feature = "fc7"
    FLAGS.projection_dim = 512
    FLAGS.train_stage_name = "stage_lstm"

    # data provider related
    FLAGS.decode_downsample_factor = 2
    FLAGS.temporal_downsample_factor = 1
    FLAGS.n_sub_frame = 150
    FLAGS.ego_previous_nstep = 30
    FLAGS.stop_future_frames = 15

    # learning related
    FLAGS.max_steps = 10000000
    FLAGS.initial_learning_rate = 0.01
    FLAGS.num_epochs_per_decay = 100
    FLAGS.learning_rate_decay_factor = 0.1

def id_config(train_dir, port):
    # experiment id related configs
    FLAGS.train_dir = train_dir
    FLAGS.tensorboard_port = port

############ Config Starts Here ####################
### code above doesn't need to change

def subsample_1(phase):
    tunable_config()

    # resources: tmux = 4
    id_config("data/car_stop_multiplier_sub1", 6200)
    if phase == "train":
        set_gpu("7")
    elif phase == "eval":
        set_gpu("7")
        FLAGS.subset = "validation"

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    FLAGS.projection_dim = 64
    # temporal don't downsample

def subsample_5(phase):
    tunable_config()

    # resources: tmux = 6
    id_config("data/car_stop_multiplier_sub5", 6201)
    if phase == "train":
        set_gpu("7")
    elif phase == "eval":
        set_gpu("4")
        FLAGS.subset = "train"

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    FLAGS.projection_dim = 64

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 30
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 3


def multi_gpu(phase):
    tunable_config()

    # resources: tmux = 6
    id_config("data/multi_gpu", 6202)
    if phase == "train":
        set_gpu("3,4")
        FLAGS.batch_size = 2
    elif phase == "eval":
        set_gpu("4")

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    FLAGS.projection_dim = 64

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 30
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 3

def dequeue_many(phase):
    tunable_config()

    # resources: tmux = 6
    id_config("data/dequeue_many", 6203)
    if phase == "train":
        set_gpu("3")
    elif phase == "eval":
        set_gpu("4")
    FLAGS.batch_size = 1

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    FLAGS.projection_dim = 64

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 30
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 3

    FLAGS.profile = False


def subsample_1_groundtruth(phase):
    tunable_config()

    # resources: tmux = 4
    id_config("data/car_stop_truth_sub1", 6200)
    if phase == "train":
        set_gpu("7")
    elif phase == "eval":
        set_gpu("7")
        FLAGS.subset = "validation"

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    FLAGS.projection_dim = 32
    # temporal don't downsample
    FLAGS.stop_future_frames = 7


# 3 new sets of experiments
def subsample_1_groundtruth_dropout(phase):
    tunable_config()

    # resources: tmux = 3
    id_config("data/car_stop_gt_dropout_sub1", 6205)
    if phase == "train":
        set_gpu("2")
    elif phase == "eval":
        set_gpu("2")
        FLAGS.subset = "validation"

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    FLAGS.projection_dim = 256
    FLAGS.stop_future_frames = 7

    # regularization related
    FLAGS.add_dropout_layer = True
    FLAGS.add_dim_reduction = True

def subsample_5_groundtruth_dropout(phase):
    tunable_config()

    # resources: tmux = 4
    id_config("data/car_stop_gt_dropout_sub5", 6204)
    if phase == "train":
        set_gpu("7")
    elif phase == "eval":
        set_gpu("7")
        FLAGS.subset = "validation"

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    #FLAGS.projection_dim = 256

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 30
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 2

    # regularization related
    FLAGS.add_dropout_layer = True
    FLAGS.add_dim_reduction = False

# stopped
def subsample_5_groundtruth_dropout_dim(phase):
    tunable_config()

    # resources: tmux = 5
    id_config("data/car_stop_gt_dropout_dim_sub5", 6206)
    if phase == "train":
        set_gpu("3")
    elif phase == "eval":
        set_gpu("3")
        FLAGS.subset = "validation"

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    FLAGS.projection_dim = 256

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 30
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 2

    # regularization related
    FLAGS.add_dropout_layer = True
    FLAGS.add_dim_reduction = True

# new set of experiments
def subsample_5_aug_drop05_nodim(phase):
    tunable_config()

    # resources: tmux = 4
    id_config("data/car_stop_aug_drop05_nodim_sub5", 6207)
    if phase == "train":
        set_gpu("2")
    elif phase == "eval":
        set_gpu("2")
        FLAGS.subset = "validation"

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    #FLAGS.projection_dim = 256

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 30
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 2

    # regularization related
    FLAGS.add_dropout_layer = True
    FLAGS.keep_prob = 0.5
    FLAGS.add_dim_reduction = False

def subsample_5_aug_drop05_dim256(phase):
    tunable_config()

    # resources: tmux = 4
    id_config("data/car_stop_aug_drop05_dim256_sub5", 6208)
    if phase == "train":
        set_gpu("3")
    elif phase == "eval":
        set_gpu("3")
        FLAGS.subset = "validation"

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    FLAGS.projection_dim = 256

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 30
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 2

    # regularization related
    FLAGS.add_dropout_layer = True
    FLAGS.keep_prob = 0.5
    FLAGS.add_dim_reduction = True

def subsample_5_aug_nodrop_dim256(phase):
    tunable_config()

    # resources: tmux = 4
    id_config("data/car_stop_aug_nodrop_dim256_sub5", 6209)
    if phase == "train":
        set_gpu("7")
    elif phase == "eval":
        set_gpu("7")
        FLAGS.subset = "validation"

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    FLAGS.projection_dim = 256

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 30
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 2

    # regularization related
    FLAGS.add_dropout_layer = False
    FLAGS.keep_prob = 0.5
    FLAGS.add_dim_reduction = True

# control experiment for huazhe
# autolabeled data with new formulation future 0.5 second
def subsample_5_huazhe_autolabel(phase):
    tunable_config()

    # resources: tmux = 4
    id_config("data/car_stop_autolabel_sub5", 6209)
    if phase == "train":
        set_gpu("2")
    elif phase == "eval":
        set_gpu("2")
        FLAGS.subset = "validation"

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    FLAGS.projection_dim = 64

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 30
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 2

    # regularization related
    FLAGS.add_dropout_layer = False
    FLAGS.keep_prob = 0.5
    FLAGS.add_dim_reduction = True

    FLAGS.data_dir = '/home/yang/code/scale_inv/tfcnn/data/nexar_100_autolabel_2'

# the dilation set of experiments
def subsample_5_aug_drop01_nodim_dilation(phase):
    tunable_config()

    # resources: tmux = 4
    id_config("data/car_stop_drop01_nodim_dilation_sub5", 6210)
    if phase == "train":
        set_gpu("2")
    elif phase == "eval":
        set_gpu("2")
        FLAGS.subset = "validation"

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    FLAGS.projection_dim = 64

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    # should be 30, but here shuffling 30s in to 3*9 (throw away 3 in the end)
    FLAGS.n_sub_frame = 6
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 2

    # regularization related
    FLAGS.add_dropout_layer = False
    #FLAGS.keep_prob = 0.1
    FLAGS.add_dim_reduction = True

    # change to dilation
    FLAGS.image_network_arch = "Dilation_front"
    FLAGS.pretrained_model_path = "/home/yang/si/data/pretrained_models/tf.dilation10.bin"
    FLAGS.cnn_feature = "fc7"

def subsample_5_aug_nodrop_dim64_dilation_conv53(phase):
    tunable_config()

    # resources: tmux = 4
    id_config("data/car_stop_nodrop_dim64_dilation_conv53", 6212)
    if phase == "train":
        set_gpu("5")
    elif phase == "eval":
        set_gpu("5")
        FLAGS.subset = "validation"

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    FLAGS.projection_dim = 64

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    # should be 30, but here shuffling 30s in to 3*9 (throw away 3 in the end)
    FLAGS.n_sub_frame = 6
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 2

    # regularization related
    FLAGS.add_dropout_layer = False
    #FLAGS.keep_prob = 0.1
    FLAGS.add_dim_reduction = True

    # change to dilation
    FLAGS.image_network_arch = "Dilation_front"
    FLAGS.pretrained_model_path = "/home/yang/si/data/pretrained_models/tf.dilation10.bin"
    FLAGS.cnn_feature = "conv5_3"

def subsample_5_aug_drop01_dim64_dilation(phase):
    tunable_config()

    # resources: tmux = 4
    id_config("data/car_stop_drop01_dim64_dilation_sub5", 6211)
    if phase == "train":
        set_gpu("4")
    elif phase == "eval":
        set_gpu("4")
        FLAGS.subset = "validation"

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    FLAGS.projection_dim = 64

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    # should be 30, but here shuffling 30s in to 3*9 (throw away 3 in the end)
    FLAGS.n_sub_frame = 6
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 2

    # regularization related
    FLAGS.add_dropout_layer = True
    FLAGS.keep_prob = 0.1
    FLAGS.add_dim_reduction = True

    # change to dilation
    FLAGS.image_network_arch = "Dilation_front"
    FLAGS.pretrained_model_path = "/home/yang/si/data/pretrained_models/tf.dilation10.bin"
    FLAGS.cnn_feature = "fc7"


# the caffenet FCN experiments
def subsample_5_caffeFCN_nodrop_dim64_fc7(phase):
    tunable_config()

    # resources: tmux = 4
    id_config("data/car_stop_subsample_5_caffeFCN_nodrop_dim64_fc7", 6211)
    if phase == "train":
        set_gpu("6")
    elif phase == "eval":
        set_gpu("6")
        FLAGS.subset = "validation"

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    FLAGS.projection_dim = 64

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 30
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 2

    # regularization related
    FLAGS.add_dropout_layer = False
    #FLAGS.keep_prob = 0.1
    FLAGS.add_dim_reduction = True

    # change to dilation
    FLAGS.image_network_arch = "CaffeNet_dilation"
    FLAGS.pretrained_model_path = "/home/yang/si/data/pretrained_models/tf.caffenet.bin"
    FLAGS.cnn_feature = "fc7"

def subsample_5_caffeFCN_nodrop_dim64_conv5(phase):
    tunable_config()

    # resources: tmux = 4
    id_config("data/car_stop_subsample_5_caffeFCN_nodrop_dim64_conv5", 6212)
    if phase == "train":
        set_gpu("7")
    elif phase == "eval":
        set_gpu("7")
        FLAGS.subset = "validation"

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    FLAGS.projection_dim = 64

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 30
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 2

    # regularization related
    FLAGS.add_dropout_layer = False
    #FLAGS.keep_prob = 0.1
    FLAGS.add_dim_reduction = True

    # change to dilation
    FLAGS.image_network_arch = "CaffeNet_dilation"
    FLAGS.pretrained_model_path = "/home/yang/si/data/pretrained_models/tf.caffenet.bin"
    FLAGS.cnn_feature = "conv5"



# experiments on large dataset
def car_stop_large_caffeFCN_nodrop_dim64_fc7(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 6211)
    set_gpu("7")
    FLAGS.batch_size = 1

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    # dim reduction
    FLAGS.add_dim_reduction = True
    FLAGS.projection_dim = 64
    # dropout
    FLAGS.add_dropout_layer = False
    FLAGS.keep_prob = 0.1

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 100
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 2

    # change to dilation
    FLAGS.image_network_arch = "CaffeNet_dilation"
    FLAGS.pretrained_model_path = "/home/yang/si/data/pretrained_models/tf.caffenet.bin"
    FLAGS.cnn_feature = "fc7"

    # some changed default behavior
    FLAGS.decode_downsample_factor = 1
    FLAGS.speed_limit_as_stop = 0.05
    FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_speed/"
    FLAGS.data_provider = "nexar_large_speed"
    FLAGS.num_epochs_per_decay = 1

def car_stop_large_caffenet_nodrop_dim64_fc7(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 6212)
    set_gpu("4")
    FLAGS.batch_size = 1

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    # dim reduction
    FLAGS.add_dim_reduction = True
    FLAGS.projection_dim = 64
    # dropout
    FLAGS.add_dropout_layer = False
    FLAGS.keep_prob = 0.1

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 100
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 2

    # change to dilation
    FLAGS.image_network_arch = "CaffeNet"
    FLAGS.pretrained_model_path = "/home/yang/si/data/pretrained_models/tf.caffenet.bin"
    FLAGS.cnn_feature = "fc7"

    # some changed default behavior
    FLAGS.decode_downsample_factor = 1
    FLAGS.speed_limit_as_stop = 0.05
    FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_speed/"
    FLAGS.data_provider = "nexar_large_speed"
    FLAGS.num_epochs_per_decay = 1

# a simple baseline
def car_stop_large_caffenet_nodrop_dim64_fc7_CNN_FC(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 6213)
    set_gpu("3")
    FLAGS.batch_size = 1

    # the parameter setting for this experiment
    #FLAGS.lstm_hidden_units = "64"
    # dim reduction
    FLAGS.add_dim_reduction = True
    FLAGS.projection_dim = 64
    # dropout
    FLAGS.add_dropout_layer = False
    FLAGS.keep_prob = 0.1

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 100
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 2

    # change to dilation
    FLAGS.image_network_arch = "CaffeNet"
    FLAGS.pretrained_model_path = "/home/yang/si/data/pretrained_models/tf.caffenet.bin"
    FLAGS.cnn_feature = "fc7"

    # some changed default behavior
    FLAGS.decode_downsample_factor = 1
    FLAGS.speed_limit_as_stop = 0.05
    FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_speed/"
    FLAGS.data_provider = "nexar_large_speed"
    FLAGS.num_epochs_per_decay = 1

    # CNN_FC specific
    FLAGS.arch_selection="CNN_FC"
    FLAGS.history_window = 9
    FLAGS.cnn_fc_hidden_units = 64


# balance sampling trail
def car_stop_large_caffenet_nodrop_dim64_fc7_balance(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 6212)
    set_gpu("3")
    FLAGS.batch_size = 1

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    # dim reduction
    FLAGS.add_dim_reduction = True
    FLAGS.projection_dim = 64
    # dropout
    FLAGS.add_dropout_layer = False
    FLAGS.keep_prob = 0.1

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 100
    FLAGS.ego_previous_nstep = 6
    FLAGS.stop_future_frames = 2

    # change to dilation
    FLAGS.image_network_arch = "CaffeNet"
    FLAGS.pretrained_model_path = "/home/yang/si/data/pretrained_models/tf.caffenet.bin"
    FLAGS.cnn_feature = "fc7"

    # some changed default behavior
    FLAGS.decode_downsample_factor = 1
    FLAGS.speed_limit_as_stop = 0.25
    FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_filter/"
    FLAGS.data_provider = "nexar_large_speed"
    FLAGS.num_epochs_per_decay = 1



# experiments to fine tune the CNN layer representation
def car_stop_large_caffeFCN_nodrop_dim64_fc7_finetune(phase):
    car_stop_large_caffeFCN_nodrop_dim64_fc7(phase)

    tag = inspect.stack()[0][3]
    id_config("data/"+tag, 6211)
    set_gpu("4,5")
    FLAGS.batch_size=2

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 45 # 108 couldn't fit into mem
    FLAGS.stop_future_frames = 2

    FLAGS.speed_limit_as_stop = 0.3
    FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_filter/"

    # fine tune learning rate setting
    FLAGS.num_epochs_per_decay = 1
    FLAGS.initial_learning_rate = 1e-4
    FLAGS.training_step_offset = 33000
    FLAGS.train_stage_name = 'stage_all'

def car_stop_large_caffenet_nodrop_dim64_fc7_finetune(phase):
    car_stop_large_caffenet_nodrop_dim64_fc7(phase)

    tag = inspect.stack()[0][3]
    id_config("data/"+tag, 6212)
    set_gpu("6")

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 108
    FLAGS.stop_future_frames = 2

    FLAGS.speed_limit_as_stop = 0.3
    FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_filter/"

    # fine tune learning rate setting
    FLAGS.num_epochs_per_decay = 5
    FLAGS.initial_learning_rate = 1e-4
    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 91436
    FLAGS.train_stage_name = 'stage_all'

def car_stop_large_caffenet_nodrop_dim64_fc7_CNN_FC_finetune(phase):
    car_stop_large_caffenet_nodrop_dim64_fc7_CNN_FC(phase)

    tag = inspect.stack()[0][3]
    id_config("data/"+tag, 6213)
    set_gpu("3")

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 108
    FLAGS.stop_future_frames = 2

    FLAGS.speed_limit_as_stop = 0.3
    FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_filter/"

    # fine tune learning rate setting
    FLAGS.num_epochs_per_decay = 5
    FLAGS.initial_learning_rate = 1e-3
    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 98259
    FLAGS.train_stage_name = 'stage_all'

# another small baseline of CNN_FC history win=2, and classic finetuning
def car_stop_large_caffenet_nodrop_dim64_fc7_CNN_FC_win2_classicFT(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 6214)
    set_gpu("6")
    FLAGS.batch_size = 1

    # the parameter setting for this experiment
    #FLAGS.lstm_hidden_units = "64"
    # dim reduction
    FLAGS.add_dim_reduction = True
    FLAGS.projection_dim = 64
    # dropout
    FLAGS.add_dropout_layer = False
    FLAGS.keep_prob = 0.1

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 108
    FLAGS.stop_future_frames = 2

    # change to dilation
    FLAGS.image_network_arch = "CaffeNet"
    FLAGS.pretrained_model_path = "/home/yang/si/data/pretrained_models/tf.caffenet.bin"
    FLAGS.cnn_feature = "fc7"

    # some changed default behavior
    FLAGS.decode_downsample_factor = 1
    FLAGS.speed_limit_as_stop = 0.3
    FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_filter/"
    FLAGS.data_provider = "nexar_large_speed"
    FLAGS.num_epochs_per_decay = 3

    # CNN_FC specific
    FLAGS.arch_selection="CNN_FC"
    FLAGS.history_window = 2
    FLAGS.cnn_fc_hidden_units = 64
    FLAGS.initial_learning_rate = 1e-2
    FLAGS.train_stage_name = 'stage_classic_finetune'



# larger learning rate sets of experiments
def car_stop_large_caffenet_nodrop_dim64_fc7_finetune_largeLR(phase):
    car_stop_large_caffenet_nodrop_dim64_fc7(phase)

    tag = inspect.stack()[0][3]
    id_config("data/"+tag, 6220)
    if phase == 'train':
        set_gpu("0,1,2,3")
        FLAGS.batch_size = 4
        FLAGS.balance_drop_prob = 0.7
    else:
        set_gpu("1")
        FLAGS.batch_size = 1
        FLAGS.balance_drop_prob = -1.0
        FLAGS.pretrained_model_checkpoint_path = \
            '/data/yang/data/car_stop_large_caffenet_nodrop_dim64_fc7_finetune_largeLR/model.ckpt-111001'
        print("running the 111k test")

    FLAGS.num_readers = 2
    FLAGS.num_preprocess_threads = 4
    FLAGS.input_queue_memory_factor = 16
    FLAGS.num_batch_join = 4

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 108
    FLAGS.stop_future_frames = 2

    FLAGS.speed_limit_as_stop = 0.3
    #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_filter/"
    #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed/"
    FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"   
    # fine tune learning rate setting
    # if you spend 1 sec on each video, then you need 8 hour for 1 epoch
    # in practice, you will spend 1.5second, that's 12 hours
    # using 2 gpus that's 6hours, using 4, that's 3 hours
    FLAGS.num_epochs_per_decay = 2
    FLAGS.initial_learning_rate = 1e-4
    FLAGS.learning_rate_decay_factor = 0.5
    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 91436
    FLAGS.train_stage_name = 'stage_all'

    FLAGS.clip_gradient_threshold = 10.0
    FLAGS.momentum = 0.99

def car_stop_large_caffeFCN_nodrop_dim64_fc7_finetune_largeLR(phase):
    car_stop_large_caffeFCN_nodrop_dim64_fc7(phase)

    tag = inspect.stack()[0][3]
    id_config("data/"+tag, 6221)
    if phase == 'train':
        set_gpu("0,1,2,3")
        FLAGS.batch_size = 4
        FLAGS.balance_drop_prob = 0.7
        #FLAGS.n_sub_frame = 45  # 108 couldn't fit into mem
    else:
        set_gpu("1")
        FLAGS.batch_size = 1
        FLAGS.balance_drop_prob = -1.0
        FLAGS.n_sub_frame = 108
        FLAGS.run_once = False
        FLAGS.eval_interval_secs = 3600
        print("FCN auc added")

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 45 # 108 couldn't fit into mem
    FLAGS.stop_future_frames = 2

    FLAGS.speed_limit_as_stop = 0.3
    #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_filter/"
    #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed/"
    FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    # fine tune learning rate setting
    FLAGS.num_epochs_per_decay = 4
    FLAGS.initial_learning_rate = 1e-4
    FLAGS.learning_rate_decay_factor = 0.5
    FLAGS.training_step_offset = 33000
    FLAGS.train_stage_name = 'stage_all'

    FLAGS.clip_gradient_threshold = 10.0
    FLAGS.momentum = 0.99

# speed only prediction
def car_stop_CNN_FC_speedOnly(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 6215)
    set_gpu("0")
    FLAGS.batch_size = 256

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 108
    FLAGS.stop_future_frames = 2

    # some changed default behavior
    FLAGS.decode_downsample_factor = 1
    FLAGS.speed_limit_as_stop = 0.3
    FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed_only/"
    FLAGS.data_provider = "nexar_large_speed"
    FLAGS.num_epochs_per_decay = 100

    # CNN_FC specific
    FLAGS.arch_selection="CNN_FC"
    FLAGS.history_window = 9
    FLAGS.cnn_fc_hidden_units = 64
    FLAGS.initial_learning_rate = 1e-2
    FLAGS.train_stage_name = 'stage_all'

    # speed only
    FLAGS.use_image_feature = False
    FLAGS.use_previous_speed_feature = True
    FLAGS.no_image_input = True
    FLAGS.examples_per_shard = 3000
    FLAGS.num_readers = 1
    FLAGS.num_preprocess_threads = 1

    FLAGS.profile = False

# what if we predict slow down or low speed, instead of complete stop?
def car_stop_cnn_slow(phase):
    tag = inspect.stack()[0][3]
    tunable_config()
    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)
    if phase == "train":
        set_gpu("0,1")
        FLAGS.batch_size = 8
        FLAGS.balance_drop_prob = 0.3
    else:
        set_gpu("0")
        FLAGS.batch_size = 4
        FLAGS.balance_drop_prob = -1.0

    FLAGS.num_readers = 2
    FLAGS.num_preprocess_threads = 4
    FLAGS.input_queue_memory_factor = 8
    FLAGS.num_batch_join = 4

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    # dim reduction
    FLAGS.add_dim_reduction = True
    FLAGS.projection_dim = 64
    # dropout
    FLAGS.add_dropout_layer = False

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.n_sub_frame = 108
    FLAGS.stop_future_frames = 6 # make sure this make sense

    # change to dilation
    FLAGS.image_network_arch = "CaffeNet"
    FLAGS.pretrained_model_path = "/data/yang/si/data/pretrained_models/tf.caffenet.bin"
    FLAGS.cnn_feature = "fc7"

    # some data provider
    FLAGS.decode_downsample_factor = 1
    FLAGS.speed_limit_as_stop = 2.0 # TODO: make sure it make sense
    #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed/"
    FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    FLAGS.data_provider = "nexar_large_speed"

    # learning rates
    FLAGS.num_epochs_per_decay = 2
    FLAGS.initial_learning_rate = 1e-3
    FLAGS.learning_rate_decay_factor = 0.5
    FLAGS.train_stage_name = 'stage_lstm'

    FLAGS.clip_gradient_threshold = 10.0
    FLAGS.momentum = 0.99

    FLAGS.profile = False
def car_stop_cnn_slow_finetune(phase):
    car_stop_cnn_slow(phase)

    tag = inspect.stack()[0][3]
    id_config("data/"+tag, 6220)
    if phase == "train":
        set_gpu("0,1,2,3")
        FLAGS.batch_size = 4
    else:
        set_gpu("0")
        FLAGS.batch_size = 1
        FLAGS.run_once = False
        FLAGS.eval_interval_secs = 3600
        print("auc added")

    FLAGS.num_epochs_per_decay = 2
    FLAGS.initial_learning_rate = 1e-4
    FLAGS.learning_rate_decay_factor = 0.5

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 153711
    FLAGS.train_stage_name = 'stage_all'

def car_stop_fcn_slow_finetune(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        set_gpu("0,1,2,3")
        FLAGS.batch_size = 4
        FLAGS.balance_drop_prob = 0.3
        FLAGS.n_sub_frame = 45  # 108 is too long
    else:
        set_gpu("1")
        FLAGS.batch_size = 1
        FLAGS.balance_drop_prob = -1.0
        FLAGS.n_sub_frame = 108

        FLAGS.run_once = False
        FLAGS.eval_interval_secs = 600
        print("eval using gpu 1")  

    # the parameter setting for this experiment
    FLAGS.lstm_hidden_units = "64"
    # dim reduction  

    FLAGS.add_dim_reduction = True
    FLAGS.projection_dim = 64
    # dropout
    FLAGS.add_dropout_layer = False

    # temporal downsample = 5
    FLAGS.temporal_downsample_factor = 5
    FLAGS.stop_future_frames = 6 # make sure this make sense

    # change to dilation
    FLAGS.image_network_arch = "CaffeNet_dilation"
    FLAGS.pretrained_model_path = "/data/yang/si/data/pretrained_models/tf.caffenet.bin"

    FLAGS.cnn_feature = "fc7"

    # some data provider
    FLAGS.decode_downsample_factor = 1
    FLAGS.speed_limit_as_stop = 2.0 # TODO: make sure it make sense
    #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed/"
    FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    FLAGS.data_provider = "nexar_large_speed"
    # learning rates
    FLAGS.num_epochs_per_decay = 4
    FLAGS.initial_learning_rate = 1e-4
    FLAGS.learning_rate_decay_factor = 0.5
    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 125001
    FLAGS.train_stage_name = 'stage_all'

    FLAGS.clip_gradient_threshold = 10.0
    FLAGS.momentum = 0.99

def car_discrete_fcn(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("0")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    else:
        set_gpu("0")
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    FLAGS.training_step_offset = 188192
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)

    # some tunable ground truth maker
    FLAGS.stop_future_frames = 4  # make sure this make sense
    FLAGS.deceleration_thres = 2

def car_continous_fcn(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # this will be trained on dgx
        set_gpu("0,1,2,3")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    else:
        set_gpu("0")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"

    FLAGS.training_step_offset = 223001
    common_continuous_settings(phase, tag,
                               isFCN=True,
                               visEval=False)

    FLAGS.stop_future_frames = 2
    FLAGS.num_epochs_per_decay = 3


def car_stop_fcn_side_slow_finetune(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # train_dir and port num
    id_config("data/"+tag, 6230)

    if phase == "train":
        set_gpu("1,2,3")
        FLAGS.batch_size = 3
        FLAGS.balance_drop_prob = 0.3
        FLAGS.n_sub_frame = 45
        #FLAGS.num_batch_join = 4
    else:
        set_gpu("3")
        FLAGS.batch_size = 1
        FLAGS.balance_drop_prob = -1.0
        FLAGS.n_sub_frame = 100

        FLAGS.run_once = False
        FLAGS.eval_interval_secs = 600
        print("eval using gpu 3")
        FLAGS.eval_method = "car_stop"
        eval_visualize = True
        if eval_visualize:
            FLAGS.output_visualizations = True
            FLAGS.subsample_factor = 1
            FLAGS.run_once = True
        else:
            FLAGS.output_visualizations = False
            FLAGS.eval_interval_secs = 1
            FLAGS.run_once = False

    FLAGS.lstm_hidden_units = "64" 
    FLAGS.add_dim_reduction = True
    FLAGS.projection_dim = 64
    FLAGS.add_dropout_layer = False
    FLAGS.city_data = 0

    FLAGS.temporal_downsample_factor = 5
    FLAGS.stop_future_frames = 6

    FLAGS.image_network_arch = "CaffeNet_dilation"
    FLAGS.pretrained_model_path = "/data/yang/data/pretrained_models/tf.caffenet.bin"
    FLAGS.cnn_feature = "fc7"

    FLAGS.decode_downsample_factor = 1
    FLAGS.speed_limit_as_stop = 2.0
    FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_side/"
    FLAGS.data_provider = "nexar_side_info"

    FLAGS.num_epochs_per_decay = 4
    FLAGS.initial_learning_rate = 1e-3
    FLAGS.learning_rate_decay_factor = 0.5

    FLAGS.training_step_offset = 188192
    FLAGS.train_stage_name = 'stage_lstm'

    FLAGS.clip_gradient_threshold = 10.0
    FLAGS.momentum =  0.99

def car_discrete_fcn_near(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose

        set_gpu("0,1,2,3")

        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    else:
        set_gpu("2")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"

    FLAGS.training_step_offset = 238900 #288174 #238900
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)

def car_continous_cnn_near_linear(phase):

    tag = inspect.stack()[0][3]
    tunable_config()

    # train_dir and port num
    id_config("data/"+tag, 6230)

    if phase == "train":
        set_gpu("1,2,3")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "eval":
        set_gpu("0")
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase=="stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"


    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195001
    common_continuous_settings(phase, tag,
                               isFCN=False,
                               visEval=False)
    # special flags for the discritization
    FLAGS.discretize_max_angle = math.pi / 9
    FLAGS.discretize_max_speed = 20.0
    FLAGS.discretize_bin_type = "linear"

def car_discrete_cnn_near_balance(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("5")
        FLAGS.balance_drop_prob = -1.0
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("6")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    FLAGS.class_balance_epsilon = 0.5
    FLAGS.class_balance_path = "data/" + tag + "/empirical_dist"

    FLAGS.training_step_offset = 195001
    common_discrete_settings(phase, tag,
                             isFCN=False,
                             visEval=False)

def car_continuous_cnn_near_linear_balance(phase):

    tag = inspect.stack()[0][3]
    tunable_config()

    # train_dir and port num
    id_config("data/"+tag, 6230)

    if phase == "train":

        # for testing purpose
        set_gpu("2,3")
        FLAGS.balance_drop_prob = -1.0
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase=="eval":

        set_gpu("0")
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    common_continuous_settings(phase, tag,
                               isFCN=False,
                               visEval=False)

    # TODO: think about the discritization
    FLAGS.class_balance_epsilon = 0.5
    FLAGS.class_balance_path = "data/" + tag + "/empirical_dist"

    # special flags for the discritization
    # TODO: think about the ranges, and linear and sigma
    FLAGS.discretize_max_angle = math.pi / 9
    FLAGS.discretize_max_speed = 20.0
    FLAGS.discretize_bin_type = "linear"
    FLAGS.discretize_label_gaussian_sigma = 1.0

def common_discrete_settings(phase, tag, isFCN, visEval):
    if phase == "train":
        FLAGS.n_sub_frame = 45 if isFCN else 108
    elif phase=="eval":
        FLAGS.balance_drop_prob = -1.0

        FLAGS.n_sub_frame = 108

        FLAGS.eval_method = "car_discrete"

        if visEval:
            FLAGS.output_visualizations=True
            FLAGS.subsample_factor=10

            FLAGS.run_once = True
        else:
            FLAGS.output_visualizations = False
            FLAGS.eval_interval_secs = 1
            FLAGS.run_once = False
    elif phase == "stat":
        set_gpu("0")
        FLAGS.subset="train"
        FLAGS.n_sub_frame = 108


        FLAGS.stat_output_path = "data/" + tag + "/empirical_dist"
        FLAGS.eval_method = "stat_labels"
        FLAGS.no_image_input = True
        FLAGS.subsample_factor = 10

    if not(phase=="board" or phase=="stat"):
        FLAGS.batch_size = 1 * FLAGS.num_gpus

    # the parameter setting for this experiment
    FLAGS.sub_arch_selection = "car_discrete"
    FLAGS.lstm_hidden_units = "64"
    # dim reduction

    FLAGS.add_dim_reduction = True
    FLAGS.projection_dim = 64
    FLAGS.add_dropout_layer = False
    
    FLAGS.decode_downsample_factor = 1
    FLAGS.temporal_downsample_factor = 5


    FLAGS.data_provider = "nexar_large_speed"
    # some tunable ground truth maker
    FLAGS.speed_limit_as_stop = 2.0  # TODO: make sure it make sense
    FLAGS.stop_future_frames = 1 # make sure this make sense
    FLAGS.deceleration_thres = 1
    FLAGS.no_slight_turn = True

    # change to dilation
    FLAGS.image_network_arch = "CaffeNet_dilation" if isFCN else "CaffeNet"
    FLAGS.pretrained_model_path = "/data/yang/si/data/pretrained_models/tf.caffenet.bin"
    FLAGS.cnn_feature = "fc7"

    # learning rates
    FLAGS.num_epochs_per_decay = 4 if isFCN else 2
    FLAGS.initial_learning_rate = 1e-4
    FLAGS.learning_rate_decay_factor = 0.5

    FLAGS.train_stage_name = 'stage_all'


    FLAGS.clip_gradient_threshold = 10.0
    FLAGS.momentum =  0.99
def car_discrete_fcn_near_city_small(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("1,2,3")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.n_sub_frame = 45  # 108 is too long
        
    else:
        set_gpu('5')
        FLAGS.balance_drop_prob = -1.0
        FLAGS.n_sub_frame = 100
        FLAGS.display_summary = 1

    common_discrete_small_settings(phase,
                                   tag,
                                   isFCN=True,
                                   visEval=False)
    FLAGS.PTrain = 1
    FLAGS.batch_size = 1 * FLAGS.num_gpus
    FLAGS.city_data = 1
    FLAGS.only_seg = 0
    FLAGS.temporal_downsample_factor = 5
    FLAGS.training_step_offset = 238900
    FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed/"
    FLAGS.FRAMES_IN_SEG = 500
    FLAGS.subsample_factor = 1
def car_discrete_only_segmentation(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("0")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.n_sub_frame = 45  # 108 is too long
        
    else:
        set_gpu("3")
        FLAGS.balance_drop_prob = -1.0
        FLAGS.n_sub_frame = 100
        
    common_discrete_small_settings(phase,
                                   tag,
                                   isFCN=True,
                                   visEval=False)

    FLAGS.batch_size = 1 * FLAGS.num_gpus
    FLAGS.city_data = 0
    FLAGS.training_step_offset = 238900
    FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_side/"
    FLAGS.only_seg = 1
    FLAGS.subsample_factor = 1
    FLAGS.FRAMES_IN_SEG = 500
    FLAGS.run_once = True
def car_discrete_fcn_near_city_baseline_small(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("1")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.n_sub_frame = 45  # 108 is too long
        
    else:
        set_gpu("0")
        FLAGS.balance_drop_prob = -1.0
        FLAGS.n_sub_frame = 100

    common_discrete_small_settings(phase,
                                   tag,
                                   isFCN=True,
                                   visEval=False)
    FLAGS.PTrain = 1
    FLAGS.batch_size = 1 * FLAGS.num_gpus
    FLAGS.city_data = 0
    FLAGS.temporal_downsample_factor = 1
    FLAGS.training_step_offset = 238900
    FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_side/"

    FLAGS.num_batch_join = 1 if phase == "eval" else 4
    FLAGS.enable_basenet_dropout = True

def common_continuous_settings(phase, tag, isFCN, visEval):
    common_discrete_settings(phase, tag, isFCN, visEval)
    if phase == "eval":
        FLAGS.eval_method = "car_continuous"
    FLAGS.sub_arch_selection = "car_loc_xy"

def car_discrete_cnn_near(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("1,2")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase=="eval":
        set_gpu("0")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 256376 #195001
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

def car_discrete_cnn_TCNN(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("0")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase=="eval":
        set_gpu("0")
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    FLAGS.training_step_offset = 195001 #256001 #195001
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 9
    FLAGS.cnn_fc_hidden_units = 64

def car_discrete_cnn_TCNN1(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("3")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("0")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    FLAGS.training_step_offset = 195001 #221840 #195001
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64

def car_discrete_cnn_speed(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("1,2,3")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data1/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("4")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195001 #245940 #195001
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=True)
    FLAGS.use_previous_speed_feature = True
    FLAGS.unique_experiment_name = tag

    #FLAGS.num_batch_join=1
    #FLAGS.num_preprocess_threads=1

def car_discrete_cnn_nobalance(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("2,3")
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase=="eval":
        set_gpu("5")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195001
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)
    FLAGS.balance_drop_prob = -1.0

def car_discrete_speed_only(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("1")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("1")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 4542 #195001
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)
    FLAGS.use_previous_speed_feature = True
    FLAGS.use_image_feature = False
    FLAGS.no_image_input = True

    FLAGS.unique_experiment_name = tag

    FLAGS.batch_size = 10 if phase == "train" else 1
    FLAGS.num_epochs_per_decay = 8
    FLAGS.num_batch_join = 1
    FLAGS.num_preprocess_threads = 1

def car_discrete_cnn_reduce_LSTM_param(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("0")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("3")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195001
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.add_hidden_layer_before_LSTM = 256
    FLAGS.unique_experiment_name = tag

def car_discrete_cnn_dropout(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("5")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("5")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195001
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=True)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True

def car_continuous_cnn_largeRange_moreBins_smoother(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("2")
        FLAGS.balance_drop_prob = -1.0
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("1")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    common_continuous_settings(phase, tag,
                               isFCN=False,
                               visEval=True)
    FLAGS.training_step_offset = 216139
    FLAGS.unique_experiment_name = tag
    FLAGS.pretrained_model_path = "/home/yang/si/data/pretrained_models/tf.caffenet.bin"

    # special flags for the discritization
    # TODO: think about the ranges, and linear and sigma
    FLAGS.discretize_max_angle = math.pi / 2 * 0.99
    FLAGS.discretize_max_speed = 30 * 0.99
    FLAGS.discretize_bin_type = "linear"
    FLAGS.discretize_n_bins = 180
    FLAGS.discretize_label_gaussian_sigma = 10.0

def car_discrete_fcn_small_dataset(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("2,3")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_side"
    elif phase=="eval":
        set_gpu("6")
        #FLAGS.data_dir = "/scratch/tfrecord_side"
        #FLAGS.data_dir = "/y/yang/tfrecord_side"
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.n_sub_frame = 100
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 238900
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = False
    FLAGS.is_small_side_info_dataset = False
    #FLAGS.num_epochs_per_decay = 0.3
    FLAGS.non_random_temporal_downsample = True
    FLAGS.run_once = True

    if phase=="eval":
        FLAGS.num_preprocess_threads=1
        FLAGS.subsample_factor = 1

def car_discrete_fcn_TCNN1(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("1,2")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
        FLAGS.data_dir = "/data1/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("0")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    FLAGS.training_step_offset = 238900
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)

    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64

def car_continuous_cnn_custom(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("1")
        FLAGS.balance_drop_prob = -1.0
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("1")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    common_continuous_settings(phase, tag,
                               isFCN=False,
                               visEval=True)
    FLAGS.training_step_offset = 216139
    FLAGS.unique_experiment_name = tag
    FLAGS.pretrained_model_path = "/home/yang/si/data/pretrained_models/tf.caffenet.bin"

    # special flags for the discritization
    FLAGS.discretize_bin_type = "custom"
    FLAGS.discretize_n_bins = 22
    FLAGS.discretize_label_gaussian_sigma = 0.5
    FLAGS.use_previous_speed_feature = True
    FLAGS.use_simplifed_continuous_vis = True

def car_continuous_cnn_custom_balance(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("2")
        FLAGS.balance_drop_prob = -1.0
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
    elif phase=="eval":
        # TODO: not evaluating yet
        set_gpu("1")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
    elif phase == "stat":
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"

    common_continuous_settings(phase, tag,
                               isFCN=False,
                               visEval=False)
    FLAGS.training_step_offset = 216139
    FLAGS.unique_experiment_name = tag
    FLAGS.pretrained_model_path = "/home/yang/si/data/pretrained_models/tf.caffenet.bin"

    # special flags for the discritization
    FLAGS.discretize_bin_type = "custom"
    FLAGS.discretize_n_bins = 22
    FLAGS.discretize_label_gaussian_sigma = 0.5
    FLAGS.use_previous_speed_feature = True

    FLAGS.class_balance_epsilon = 0.5
    FLAGS.class_balance_path = "data/" + tag + "/empirical_dist"

def car_continuous_cnn_custom_balance_future(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("1")
        FLAGS.balance_drop_prob = -1.0
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("6")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"

    common_continuous_settings(phase, tag,
                               isFCN=False,
                               visEval=True)
    FLAGS.training_step_offset = 216139
    FLAGS.unique_experiment_name = tag
    FLAGS.pretrained_model_path = "/data/yang/si/data/pretrained_models/tf.caffenet.bin"

    # special flags for the discritization
    FLAGS.discretize_bin_type = "custom"
    FLAGS.discretize_n_bins = 22
    FLAGS.discretize_label_gaussian_sigma = 0.5
    FLAGS.use_previous_speed_feature = True

    FLAGS.class_balance_epsilon = 0.5
    FLAGS.class_balance_path = "data/" + tag + "/empirical_dist"

    FLAGS.stop_future_frames = 6
    if phase == "stat":
        FLAGS.subsample_factor = 30
    FLAGS.use_simplifed_continuous_vis = True

def car_discrete_fcn_near_cnn_init(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("0,1,2,3")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    else:
        set_gpu("0")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data1/yang/tfrecord_fix_speed"

    FLAGS.training_step_offset = 195001 #238900 #288174 #238900
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)
    FLAGS.unique_experiment_name = tag

def car_discrete_fcn_near_dropout(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("1,2,3")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    else:
        set_gpu("0")
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data1/yang/tfrecord_fix_speed"

    FLAGS.training_step_offset = 195001 #238900 #288174 #238900
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=True)
    FLAGS.unique_experiment_name = tag
    FLAGS.dropout_LSTM_keep_prob = 0.1
    FLAGS.use_simplifed_continuous_vis = True

def car_discrete_fcn_near_dropout_basedrop(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_fcn_near_dropout_basedrop
    # ssh -N -L 7228:localhost:7228 leviathan.ist.berkeley.edu &
    # open tensorboard on browser and record experiment on the excel

    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 7228)

    if phase == "train":
        # for testing purpose
        set_gpu("3")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "eval":
        set_gpu("3")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data1/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "board":
        set_gpu("0")

    FLAGS.training_step_offset = 195117 #238900 #288174 #238900
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)
    FLAGS.unique_experiment_name = tag
    FLAGS.dropout_LSTM_keep_prob = 0.1
    #FLAGS.use_simplifed_continuous_vis = True
    FLAGS.enable_basenet_dropout = True


def car_continuous_cnn_paper_linear(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("1")
        FLAGS.balance_drop_prob = -1.0
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("0")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    common_continuous_settings(phase, tag,
                               isFCN=False,
                               visEval=True)
    FLAGS.class_balance_epsilon = 0.5
    FLAGS.class_balance_path = "data/" + tag + "/empirical_dist"

    FLAGS.training_step_offset = 195001
    FLAGS.unique_experiment_name = tag
    FLAGS.pretrained_model_path = "/data/yang/si/data/pretrained_models/tf.caffenet.bin"

    # special flags for the discritization
    # TODO: think about the ranges, and linear and sigma
    FLAGS.discretize_max_angle = math.pi / 2 * 0.99
    FLAGS.discretize_max_speed = 30 * 0.99
    FLAGS.discretize_bin_type = "linear"
    FLAGS.discretize_n_bins = 180
    FLAGS.discretize_label_gaussian_sigma = 0.5
    if phase == "stat":
        FLAGS.subsample_factor = 30
    elif phase == "eval":
        FLAGS.pdf_normalize_bins = False
        FLAGS.use_simplifed_continuous_vis = True

def car_continuous_cnn_paper_log(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("2")
        FLAGS.balance_drop_prob = -1.0
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("2")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    common_continuous_settings(phase, tag,
                               isFCN=False,
                               visEval=False)
    FLAGS.class_balance_epsilon = 0.5
    FLAGS.class_balance_path = "data/" + tag + "/empirical_dist"

    FLAGS.training_step_offset = 195001
    FLAGS.unique_experiment_name = tag
    FLAGS.pretrained_model_path = "/data/yang/si/data/pretrained_models/tf.caffenet.bin"

    # special flags for the discritization
    # TODO: think about the ranges, and linear and sigma
    FLAGS.discretize_max_angle = math.pi / 2 * 0.99
    FLAGS.discretize_max_speed = 30 * 0.99
    FLAGS.discretize_bin_type = "log"
    FLAGS.discretize_n_bins = 21
    FLAGS.discretize_label_gaussian_sigma = 0.5
    if phase == "stat":
        FLAGS.subsample_factor = 30

def car_continuous_cnn_paper_custom_balance(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("3")
        FLAGS.balance_drop_prob = -1.0
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("4")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    common_continuous_settings(phase, tag,
                               isFCN=False,
                               visEval=True)
    FLAGS.class_balance_epsilon = 0.5
    FLAGS.class_balance_path = "data/" + tag + "/empirical_dist"

    FLAGS.training_step_offset = 195001
    FLAGS.unique_experiment_name = tag
    FLAGS.pretrained_model_path = "/data/yang/si/data/pretrained_models/tf.caffenet.bin"

    # special flags for the discritization
    FLAGS.discretize_bin_type = "custom"
    FLAGS.discretize_n_bins = 22
    FLAGS.discretize_label_gaussian_sigma = 0.5
    if phase == "stat":
        FLAGS.subsample_factor = 30
    elif phase == "eval":
        FLAGS.pdf_normalize_bins = False
        FLAGS.use_simplifed_continuous_vis = True

def car_continuous_cnn_paper_custom_nobalance(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("1")
        FLAGS.balance_drop_prob = -1.0
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data1/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("4")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    common_continuous_settings(phase, tag,
                               isFCN=False,
                               visEval=True)

    FLAGS.training_step_offset = 195001
    FLAGS.unique_experiment_name = tag
    FLAGS.pretrained_model_path = "/data/yang/si/data/pretrained_models/tf.caffenet.bin"

    # special flags for the discritization
    FLAGS.discretize_bin_type = "custom"
    FLAGS.discretize_n_bins = 22
    FLAGS.discretize_label_gaussian_sigma = 0.5
    if phase == "stat":
        FLAGS.subsample_factor = 30
    elif phase == "eval":
        FLAGS.pdf_normalize_bins = False
        FLAGS.use_simplifed_continuous_vis = True

def car_continuous_cnn_paper_custom_extremeBalance(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("2")
        FLAGS.balance_drop_prob = -1.0
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data1/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("5")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    common_continuous_settings(phase, tag,
                               isFCN=False,
                               visEval=False)
    FLAGS.class_balance_epsilon = 0.8
    FLAGS.class_balance_path = "data/" + tag + "/empirical_dist"

    FLAGS.training_step_offset = 195001
    FLAGS.unique_experiment_name = tag
    FLAGS.pretrained_model_path = "/data/yang/si/data/pretrained_models/tf.caffenet.bin"

    # special flags for the discritization
    FLAGS.discretize_bin_type = "custom"
    FLAGS.discretize_n_bins = 22
    FLAGS.discretize_label_gaussian_sigma = 0.5
    if phase == "stat":
        FLAGS.subsample_factor = 30

def car_discrete_fcn_TCNN1_dropout(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("0,1,2,3")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data1/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("0")
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    FLAGS.training_step_offset = 195001
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)

    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64

    FLAGS.dropout_LSTM_keep_prob = 0.5

def car_discrete_cnn_avepool(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 6230)

    if phase == "train":
        # for testing purpose
        set_gpu("0,1,2")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("2")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True

def car_discrete_cnn_avepool_no_dim_reduction(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 7222)

    if phase == "train":
        # for testing purpose
        set_gpu("3,4,5")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("3")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

def car_discrete_fcn_avepool_no_dim_reduction(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_fcn_avepool_no_dim_reduction
    # ssh -N -L 7223:localhost:7223 leviathan.ist.berkeley.edu &

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 7223)

    if phase == "train":
        # for testing purpose
        set_gpu("0,1,2")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("2")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    # changed after training for a while
    FLAGS.num_epochs_per_decay = 10

def car_discrete_fcn_avepool_no_padding(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_fcn_avepool_no_padding
    # ssh -N -L 7237:localhost:7237 leviathan.ist.berkeley.edu &
    # open tensorboard and record on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 7237)

    if phase == "train":
        # for testing purpose
        set_gpu("2,3")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("0")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    FLAGS.image_preprocess_pad = 0
    # changed after training for a while
    FLAGS.num_epochs_per_decay = 10

def car_discrete_cnn_TCNN_basenet_dropout_feature_dropout(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_cnn_TCNN_basenet_dropout_feature_dropout
    # ssh -N -L 7224:localhost:7224 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 7224)

    if phase == "train":
        # for testing purpose
        set_gpu("4,5")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("3")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    FLAGS.training_step_offset = 195117 #256001 #195001
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 9
    FLAGS.cnn_fc_hidden_units = 64

    FLAGS.dropout_LSTM_keep_prob = 0.1
    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True

def car_discrete_cnn_TCNN1_basenet_dropout(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py train car_discrete_cnn_TCNN1_basenet_dropout
    # ssh -N -L 7225:localhost:7225 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 7225)

    if phase == "train":
        # for testing purpose
        set_gpu("0,1,2")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("6")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    FLAGS.training_step_offset = 195117 #256001 #195001
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64

    #FLAGS.dropout_LSTM_keep_prob = 0.1
    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True

def car_discrete_cnn_TCNN_basenet_dropout_feature_dropout05(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_cnn_TCNN_basenet_dropout_feature_dropout05
    # ssh -N -L 7226:localhost:7226 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 7226)

    if phase == "train":
        # for testing purpose
        set_gpu("0,1,2")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("3")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    FLAGS.training_step_offset = 195117 #256001 #195001
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 9
    FLAGS.cnn_fc_hidden_units = 64

    FLAGS.dropout_LSTM_keep_prob = 0.5
    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True

def car_discrete_cnn_low(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 7222)

    if phase == "train":
        # for testing purpose
        set_gpu("0,1")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_low/"
    elif phase=="eval":
        set_gpu("0")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_low/"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    #FLAGS.add_avepool_after_dim_reduction = True
    #FLAGS.add_dim_reduction = False

    #for low res flags
    FLAGS.IM_WIDTH = 384
    FLAGS.IM_HEIGHT = 216
    FLAGS.temporal_downsample_factor = 1
    FLAGS.n_sub_frame = 110
    FLAGS.stop_future_frames = 2# TODO: 1 is next or current?
    FLAGS.FRAMES_IN_SEG = 110
    FLAGS.frame_rate = 1
    FLAGS.low_res = True
    FLAGS.batch_size = 3 * FLAGS.num_gpus

def car_discrete_cnn_low_small(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 7777)

    if phase == "train":
        # for testing purpose
        set_gpu("0,1")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_low_new/"
        #FLAGS.data_dir = "/scratch/tfrecord_low_new/"
    elif phase=="eval":
        set_gpu("2")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_low_new/"
        #FLAGS.data_dir = "/scratch/tfrecord_low_new/"
    elif phase == "stat":
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/scratch/tfrecord_low_new/"
        set_gpu("5")
        
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 241625#195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)
    #FLAGS.subset="train"
    FLAGS.stat_output_path = "data/" + tag + "/empirical_dist"
    #FLAGS.eval_method = "stat_labels"
    #FLAGS.no_image_input = True
    #FLAGS.subsample_factor = 2
    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    # for low res flags
    common_lowres_settings(phase)
    FLAGS.train_filename = 'train_small.txt'
    #for small size data
    FLAGS.num_epochs_per_decay = 10
    FLAGS.batch_size = FLAGS.num_gpus
    #FLAGS.train_stage_name = "stage_lstm"

def car_discrete_cnn_low_medium(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 7778)

    if phase == "train":
        # for testing purpose
        set_gpu("0,1,2")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_low_new/"
        #FLAGS.data_dir = "/scratch/tfrecord_low_new/"
    elif phase=="eval":
        set_gpu("5")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_low_new/"
        #FLAGS.data_dir = "/scratch/tfrecord_low_new/"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 350198
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False
    # change to average pool(done)

    #for low res flags
    common_lowres_settings(phase)
    FLAGS.train_filename = 'train_medium.txt'
    
    # decay for medium
    FLAGS.num_epochs_per_decay = 2
    FLAGS.initial_learning_rate = 5e-5
    FLAGS.learning_rate_decay_factor = 0.1



def car_discrete_fcn_near_dropout_basedrop_ptrain(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_fcn_near_dropout_basedrop
    # ssh -N -L 7228:localhost:7228 leviathan.ist.berkeley.edu &
    # open tensorboard on browser and record experiment on the excel

    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 7228)
    FLAGS.city_data = 1
    if phase == "train":
        # for testing purpose
        set_gpu("0,1")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.n_sub_frame = 45
    elif phase == "eval":
        set_gpu("1")
        FLAGS.city_data = 0
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data1/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "board":
        set_gpu("0")

    FLAGS.training_step_offset = 345566 #238900 #288174 #238900
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)
    FLAGS.unique_experiment_name = tag
    FLAGS.dropout_LSTM_keep_prob = 0.1
    #FLAGS.use_simplifed_continuous_vis = True
    FLAGS.enable_basenet_dropout = True

    FLAGS.PTrain = 1
    FLAGS.batch_size = 1 * FLAGS.num_gpus
    
    FLAGS.only_seg = 0
    FLAGS.temporal_downsample_factor = 5
    #FLAGS.training_step_offset = 238900
    FLAGS.FRAMES_IN_SEG = 540
    FLAGS.subsample_factor = 1


def car_discrete_cnn_TCNN1_basenet_dropout_avepool(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_cnn_TCNN1_basenet_dropout_avepool
    # ssh -N -L 7230:localhost:7230 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7230)

    if phase == "train":
        # for testing purpose
        set_gpu("6,7")
        FLAGS.balance_drop_prob = 0.3
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("3")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    FLAGS.training_step_offset = 195117  # 256001 #195001
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

def car_discrete_cnn_TCNN1_basenet_dropout_avepool_largeLR(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_cnn_TCNN1_basenet_dropout_avepool_largeLR
    # ssh -N -L 7232:localhost:7232 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7232)

    if phase == "train":
        # for testing purpose
        set_gpu("0,1")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("2")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    FLAGS.training_step_offset = 195117  # 256001 #195001
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    # slower decay of the learning rate
    FLAGS.num_epochs_per_decay = 6


def car_discrete_cnn_avepool_no_dim_reduction_largeLR_noBalanceDrop(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_cnn_avepool_no_dim_reduction_largeLR_noBalanceDrop
    # ssh -N -L 7231:localhost:7231 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7231)

    if phase == "train":
        # for testing purpose
        set_gpu("4,5")
        # FLAGS.balance_drop_prob = 0.3
        FLAGS.balance_drop_prob = -1.0
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("2")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    FLAGS.num_epochs_per_decay = 6


def car_discrete_cnn_avepool_no_dim_reduction_largeLR(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py eval car_discrete_cnn_avepool_no_dim_reduction_largeLR
    # ssh -N -L 7229:localhost:7229 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7229)

    if phase == "train":
        # for testing purpose
        set_gpu("4,5")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("3")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=True)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    FLAGS.num_epochs_per_decay = 6

def car_discrete_cnn_avepool_no_dim_reduction_largeLR_stage(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_cnn_avepool_no_dim_reduction_largeLR_stage
    # ssh -N -L 7238:localhost:7238 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7238)

    if phase == "train":
        # for testing purpose
        set_gpu("0")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("1")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 394000 #195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    FLAGS.num_epochs_per_decay = 4
    FLAGS.train_stage_name = "stage_all" # previously stage_lstm
    FLAGS.num_batch_join = 8

def car_discrete_fcn_avepool_stride8(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_fcn_avepool_stride8
    # ssh -N -L 7233:localhost:7233 leviathan.ist.berkeley.edu &

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 7233)

    if phase == "train":
        # for testing purpose
        set_gpu("4,5")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("3")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    #FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False
    FLAGS.add_avepool_after_dim_reduction_with_stride = 8

    # changed after training for a while
    FLAGS.num_epochs_per_decay = 8

def car_discrete_fcn_avepool_stride8_largeInitLR(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_fcn_avepool_stride8_largeInitLR
    # ssh -N -L 7236:localhost:7236 leviathan.ist.berkeley.edu &

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 7236)

    if phase == "train":
        # for testing purpose
        set_gpu("3")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("3")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    #FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False
    FLAGS.add_avepool_after_dim_reduction_with_stride = 8

    # changed after training for a while
    FLAGS.num_epochs_per_decay = 8
    FLAGS.initial_learning_rate = 1e-3


def car_discrete_cnn_ConvLSTM(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_cnn_ConvLSTM
    # ssh -N -L 7234:localhost:7234 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7234)

    if phase == "train":
        # for testing purpose
        set_gpu("2,3")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("2")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True

    # reduce to 512 before ConvLSTM
    FLAGS.add_dim_reduction = True
    FLAGS.projection_dim = 512

    # conv lstm specific
    # This config try to mimic car_discrete_cnn_avepool_no_dim_reduction_largeLR maximally
    # Both in terms of the # of parameters and the output shape, while meantime trying to be "reasonable" about the
    # ConvLSTM architecture. This specific config here, do the global max (average) pool after LSTM, intead of before
    FLAGS.temporal_net = "ConvLSTM"
    FLAGS.lstm_hidden_units = "64"
    FLAGS.conv_lstm_filter_sizes = "3"
    FLAGS.conv_lstm_max_pool_factors = "5"

    FLAGS.num_epochs_per_decay = 6

def car_discrete_cnn_ConvLSTM_128(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_cnn_ConvLSTM_128
    # ssh -N -L 7239:localhost:7239 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7239)

    if phase == "train":
        # for testing purpose
        set_gpu("1")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("2")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True

    # reduce to 512 before ConvLSTM
    FLAGS.add_dim_reduction = True
    FLAGS.projection_dim = 128

    # conv lstm specific
    # This config try to mimic car_discrete_cnn_avepool_no_dim_reduction_largeLR maximally
    # Both in terms of the # of parameters and the output shape, while meantime trying to be "reasonable" about the
    # ConvLSTM architecture. This specific config here, do the global max (average) pool after LSTM, intead of before
    FLAGS.temporal_net = "ConvLSTM"
    FLAGS.lstm_hidden_units = "64"
    FLAGS.conv_lstm_filter_sizes = "3"
    FLAGS.conv_lstm_max_pool_factors = "5"

    FLAGS.num_epochs_per_decay = 6

def car_discrete_cnn_ConvLSTM_restore_channel(phase):
    # TODO: finalize this configuration
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py train car_discrete_cnn_ConvLSTM_restore_channel
    # ssh -N -L 7235:localhost:7235 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7235)

    if phase == "train":
        # for testing purpose
        set_gpu("4,5")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("4")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_dim_reduction = False

    # conv lstm specific
    # This config try to mimic car_discrete_cnn_avepool_no_dim_reduction_largeLR maximally
    # Both in terms of the # of parameters and the output shape, while meantime trying to be "reasonable" about the
    # ConvLSTM architecture. This specific config here, do the global max (average) pool after LSTM, intead of before
    FLAGS.temporal_net = "ConvLSTM"
    FLAGS.lstm_hidden_units = "64,512"
    FLAGS.conv_lstm_filter_sizes = "3,3"
    FLAGS.conv_lstm_max_pool_factors = "1,5"

    FLAGS.num_epochs_per_decay = 6

def car_discrete_cnn_avepool_no_dim_reduction_inst_low(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 7780)

    if phase == "train":
        # for testing purpose
        set_gpu("2")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_low_new/"
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed/"
    elif phase=="eval":
        set_gpu("5")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed/"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    FLAGS.FRAMES_IN_SEG = 540
    FLAGS.temporal_downsample_factor = 15
    FLAGS.n_sub_frame = 36

def car_discrete_fcn_near_dropout_basedrop_ptrain_split(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_fcn_near_dropout_basedrop
    # ssh -N -L 7228:localhost:7228 leviathan.ist.berkeley.edu &
    # open tensorboard on browser and record experiment on the excel

    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 7781)
    FLAGS.city_data = 1
    if phase == "train":
        # for testing purpose
        set_gpu("2,3")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.n_sub_frame = 20
    elif phase == "eval":
        set_gpu("1")
        FLAGS.city_data = 0
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data1/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
        FLAGS.n_sub_frame = 20
    elif phase == "board":
        set_gpu("0")

    FLAGS.training_step_offset = 345566 #238900 #288174 #238900
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)
    FLAGS.unique_experiment_name = tag
    FLAGS.dropout_LSTM_keep_prob = 0.1
    #FLAGS.use_simplifed_continuous_vis = True
    FLAGS.enable_basenet_dropout = True

    FLAGS.PTrain = 1
    FLAGS.batch_size = 1 * FLAGS.num_gpus

    FLAGS.only_seg = 0
    FLAGS.temporal_downsample_factor = 5
    #FLAGS.training_step_offset = 238900
    FLAGS.FRAMES_IN_SEG = 540
    FLAGS.subsample_factor = 1

    FLAGS.early_split = True
    FLAGS.cnn_split = 'conv4'

def car_discrete_cnn_avepool_no_dim_reduction_inst_low_res(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 7782)

    if phase == "train":
        # for testing purpose
        set_gpu("4,5")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_low_new/"
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed/"
    elif phase=="eval":
        set_gpu("5")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed/"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=True)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    FLAGS.FRAMES_IN_SEG = 540
    FLAGS.temporal_downsample_factor = 15
    FLAGS.n_sub_frame = 36
    FLAGS.image_downsample = True

def car_discrete_cnn_low_small_one_batch(phase):
    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 7783)

    if phase == "train":
        # for testing purpose
        set_gpu("0,1")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_low_new/"
        #FLAGS.data_dir = "/scratch/tfrecord_low_new/"
    elif phase=="eval":
        set_gpu("2")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        #FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_low_new/"
        #FLAGS.data_dir = "/scratch/tfrecord_low_new/"
    elif phase == "stat":
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        FLAGS.data_dir = "/scratch/tfrecord_low_new/"
        set_gpu("5")

    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117#241625
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)
    #FLAGS.subset="train"
    FLAGS.stat_output_path = "data/" + tag + "/empirical_dist"
    #FLAGS.eval_method = "stat_labels"
    #FLAGS.no_image_input = True
    #FLAGS.subsample_factor = 2
    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    # for low res flags
    common_lowres_settings(phase)
    FLAGS.train_filename = 'train_small.txt'
    #for small size data
    FLAGS.num_epochs_per_decay = 10
    FLAGS.batch_size = FLAGS.num_gpus
    #FLAGS.train_stage_name = "stage_lstm"

def car_discrete_cnn_avepool_no_dim_reduction_largeLR_release(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py eval car_discrete_cnn_avepool_no_dim_reduction_largeLR
    # ssh -N -L 7229:localhost:7229 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7229)

    if phase == "train":
        # for testing purpose
        set_gpu("6,7")
        FLAGS.balance_drop_prob = 0.3
        FLAGS.data_dir = "/data/nx-bdd-20170329/tfrecord_20170329/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("4")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20170329/tfrecord_20170329/"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("4")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    FLAGS.num_epochs_per_decay = 6

    FLAGS.release_batch = True
    FLAGS.pretrained_model_path = "/backup/yang/si/data/pretrained_models/tf.caffenet.bin"


def car_discrete_fcn_avepool_fisher_padding(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py eval car_discrete_fcn_avepool_fisher_padding
    # ssh -N -L 7240:localhost:7240 leviathan.ist.berkeley.edu &
    # open tensorboard and record on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/"+tag, 7240)

    if phase == "train":
        # for testing purpose
        set_gpu("0,1,2,3")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("4")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    # changed after training for a while
    FLAGS.num_epochs_per_decay = 10

    # change to dilation
    FLAGS.image_network_arch = "CaffeNet_padding"


def car_discrete_cnn_224_224(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py train car_discrete_cnn_224_224
    # ssh -N -L 7241:localhost:7241 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7241)

    if phase == "train":
        # for testing purpose
        set_gpu("1")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("4")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    FLAGS.num_epochs_per_decay = 2

    # specific setting to the 224 224 model
    FLAGS.resize_images = "228,228"
    if phase == "train":
        FLAGS.batch_size = 4 * FLAGS.num_gpus
        FLAGS.num_readers = 4
        FLAGS.num_preprocess_threads = 8
        FLAGS.num_batch_join = 8

def car_discrete_cnn_224_398(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py train car_discrete_cnn_224_398
    # ssh -N -L 7242:localhost:7242 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7242)

    if phase == "train":
        # for testing purpose
        set_gpu("4")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("4")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    FLAGS.num_epochs_per_decay = 2

    # specific setting to the 224 224 model
    FLAGS.resize_images = "228,405"
    if phase == "train":
        FLAGS.batch_size = 1 * FLAGS.num_gpus
        FLAGS.num_readers = 4
        FLAGS.num_preprocess_threads = 8
        FLAGS.num_batch_join = 8

def car_discrete_cnn_224_398_fcn(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py train car_discrete_cnn_224_398_fcn
    # ssh -N -L 7244:localhost:7244 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7244)

    if phase == "train":
        # for testing purpose
        set_gpu("5,6")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("5")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    FLAGS.num_epochs_per_decay = 2

    # specific setting to the 224 224 model
    FLAGS.resize_images = "228,405"
    if phase == "train":
        FLAGS.batch_size = 1 * FLAGS.num_gpus
        FLAGS.num_readers = 4
        FLAGS.num_preprocess_threads = 8
        FLAGS.num_batch_join = 8
    # special setting for the smaller FCN
    FLAGS.n_sub_frame = 108
    FLAGS.image_preprocess_pad = 0


def car_discrete_cnn_224_398_original(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py board car_discrete_cnn_224_398_original
    # ssh -N -L 7243:localhost:7243 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7243)

    if phase == "train":
        # for testing purpose
        set_gpu("3")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("5")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 0
    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    FLAGS.num_epochs_per_decay = 2

    # specific setting to the 224 224 model
    FLAGS.resize_images = "228,405"
    if phase == "train":
        FLAGS.batch_size = 1 * FLAGS.num_gpus
        FLAGS.num_readers = 4
        FLAGS.num_preprocess_threads = 8
        FLAGS.num_batch_join = 8


def car_discrete_cnn_dropout_512(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py train car_discrete_cnn_dropout_512
    # ssh -N -L 7245:localhost:7245 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 7245)

    if phase == "train":
        # for testing purpose
        set_gpu("7")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("0")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 195117
    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.projection_dim = 512

    FLAGS.num_epochs_per_decay = 2

    # specific setting to the 224 224 model
    FLAGS.resize_images = "228,405"
    if phase == "train":
        FLAGS.batch_size = 2 * FLAGS.num_gpus
        FLAGS.num_readers = 4
        FLAGS.num_preprocess_threads = 8
        FLAGS.num_batch_join = 4

    # special setting for the smaller FCN
    FLAGS.n_sub_frame = 108
    FLAGS.image_preprocess_pad = 0

def car_discrete_cnn_dropout_512_stage(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py train car_discrete_cnn_dropout_512_stage
    # ssh -N -L 7246:localhost:7246 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    # changes 3: gpu, batch, data_dir
    id_config("data/"+tag, 7246)

    if phase == "train":
        # for testing purpose
        set_gpu("7")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase=="eval":
        set_gpu("0")
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        #FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    common_discrete_settings(phase,
                             tag,
                             isFCN=True,
                             visEval=False)

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.projection_dim = 512

    FLAGS.num_epochs_per_decay = 2

    # specific setting to the 224 224 model
    FLAGS.resize_images = "228,405"
    if phase == "train":
        FLAGS.batch_size = 2 * FLAGS.num_gpus
        FLAGS.num_readers = 4
        FLAGS.num_preprocess_threads = 8
        FLAGS.num_batch_join = 4

    # special setting for the smaller FCN
    FLAGS.n_sub_frame = 108
    FLAGS.image_preprocess_pad = 0

    FLAGS.train_stage_name = "stage_all"
    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 292001

def car_discrete_cnn_224_224_8s(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py eval car_discrete_cnn_224_224_8s
    # ssh -N -L 7247:localhost:7247 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7247)

    if phase == "train":
        # for testing purpose
        set_gpu("4")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("2")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    # modify to use the 8s setting
    FLAGS.n_sub_frame = 108
    FLAGS.image_network_arch = "CaffeNet_dilation8"
    FLAGS.num_epochs_per_decay = 2
    FLAGS.image_preprocess_pad = 0
    FLAGS.train_stage_name = "stage_all"
    # make dir and copy the checkpoint before finetuning
    #FLAGS.training_step_offset = 217001
    FLAGS.training_step_offset = 278000


    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = True
    FLAGS.add_dim_reduction = False

    # specific setting to the 224 224 model
    FLAGS.resize_images = "228,228"
    if phase == "train":
        FLAGS.batch_size = 2 * FLAGS.num_gpus
        FLAGS.num_readers = 4
        FLAGS.num_preprocess_threads = 8
        FLAGS.num_batch_join = 4

def car_discrete_cnn_224_224_8s_ConvLSTM512(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py train car_discrete_cnn_224_224_8s_ConvLSTM512
    # ssh -N -L 7248:localhost:7248 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7248)

    if phase == "train":
        # for testing purpose
        set_gpu("2,3")
        FLAGS.balance_drop_prob = 0.3
        #FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("1")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("0")

    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 217001

    # modify to use the 8s setting
    FLAGS.n_sub_frame = 108
    FLAGS.image_network_arch = "CaffeNet_dilation8"
    FLAGS.num_epochs_per_decay = 2
    FLAGS.image_preprocess_pad = 0
    FLAGS.train_stage_name = "stage_all"


    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True

    # specific setting to the 224 224 model
    FLAGS.resize_images = "228,228"
    if phase == "train":
        FLAGS.batch_size = 1 * FLAGS.num_gpus
        FLAGS.num_readers = 4
        FLAGS.num_preprocess_threads = 8
        FLAGS.num_batch_join = 4

    # reduce to 512 before ConvLSTM
    FLAGS.add_dim_reduction = True
    FLAGS.projection_dim = 512

    # conv lstm specific
    # This config try to mimic car_discrete_cnn_avepool_no_dim_reduction_largeLR maximally
    # Both in terms of the # of parameters and the output shape, while meantime trying to be "reasonable" about the
    # ConvLSTM architecture. This specific config here, do the global max (average) pool after LSTM, intead of before
    FLAGS.temporal_net = "ConvLSTM"
    FLAGS.lstm_hidden_units = "64"
    FLAGS.conv_lstm_filter_sizes = "3"
    FLAGS.conv_lstm_max_pool_factors = "1"

    # add a drop out layer
    FLAGS.add_dropout_layer = True
    FLAGS.keep_prob = 0.5


def car_discrete_cnn_224_224_8s_priv_dim_reduction(phase):
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py eval car_discrete_cnn_224_224_8s_priv_dim_reduction
    # ssh -N -L 7249:localhost:7249 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7249)

    if phase == "train":
        # for testing purpose
        set_gpu("0,1")
        FLAGS.balance_drop_prob = 0.3
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("3")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("7")

    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 359000 #315001

    # modify to use the 8s setting
    FLAGS.n_sub_frame = 108
    FLAGS.image_network_arch = "CaffeNet"#"CaffeNet_dilation"
    FLAGS.segmentation_network_arch = "CaffeNet_dilation8"
    FLAGS.num_epochs_per_decay = 2
    FLAGS.image_preprocess_pad = -1
    FLAGS.train_stage_name = "stage_all" #"stage_lstm"

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = False #True
    FLAGS.add_dim_reduction = True #False

    # specific setting to the 224 224 model
    FLAGS.resize_images = "228,228"
    FLAGS.city_data = 1
    if phase == "train":
        FLAGS.city_image_list = '/backup/BDDNexar/Harry_config/Color_train_harry.txt'
        FLAGS.city_label_list = '/backup/BDDNexar/Harry_config/TrainLabels_train_harry.txt'
    elif phase == "eval":
        FLAGS.city_image_list = '/backup/BDDNexar/Harry_config/Color_val_harry.txt'
        FLAGS.city_label_list = '/backup/BDDNexar/Harry_config/TrainLabels_val_harry.txt'

    FLAGS.early_split = True
    FLAGS.cnn_split = 'drop6'

    if phase == "train":
        FLAGS.batch_size = 1 * FLAGS.num_gpus
        FLAGS.num_readers = 4
        FLAGS.num_preprocess_threads = 8
        FLAGS.num_batch_join = 4

def car_discrete_cnn_224_224_8s_priv_dim_reduction_fc7(phase):
    # TODO: this is not up to date version
    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py eval car_discrete_cnn_224_224_8s_priv_dim_reduction_fc7
    # ssh -N -L 7250:localhost:7250 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7250)

    if phase == "train":
        # for testing purpose
        set_gpu("1")
        FLAGS.balance_drop_prob = 0.3
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("0")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("7")

    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 315001

    # modify to use the 8s setting
    FLAGS.n_sub_frame = 108
    FLAGS.image_network_arch = "CaffeNet"#"CaffeNet_dilation"
    FLAGS.segmentation_network_arch = "CaffeNet_dilation8"
    FLAGS.num_epochs_per_decay = 2
    FLAGS.image_preprocess_pad = -1
    FLAGS.train_stage_name = "stage_lstm"

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_avepool_after_dim_reduction = False #True
    FLAGS.add_dim_reduction = True #False

    # specific setting to the 224 224 model
    FLAGS.resize_images = "228,228"
    FLAGS.city_data = 1
    if phase == "train":
        FLAGS.city_image_list = '/backup/BDDNexar/Harry_config/Color_train_harry.txt'
        FLAGS.city_label_list = '/backup/BDDNexar/Harry_config/TrainLabels_train_harry.txt'
    elif phase == "eval":
        FLAGS.city_image_list = '/backup/BDDNexar/Harry_config/Color_val_harry.txt'
        FLAGS.city_label_list = '/backup/BDDNexar/Harry_config/TrainLabels_val_harry.txt'

    FLAGS.early_split = True
    FLAGS.cnn_split = 'fc7'

    if phase == "train":
        FLAGS.batch_size = 1 * FLAGS.num_gpus
        FLAGS.num_readers = 4
        FLAGS.num_preprocess_threads = 8
        FLAGS.num_batch_join = 4

def car_discrete_fcn_dim_reduction_nopad(phase):
    # should be init from car_discrete_fcn_near_dropout

    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py train car_discrete_fcn_dim_reduction_nopad
    # ssh -N -L 7251:localhost:7251 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7251)

    if phase == "train":
        # for testing purpose
        set_gpu("0,4")
        FLAGS.balance_drop_prob = 0.3
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("2")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("7")

    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 242000 #211910
    # stage one reached 75.67 but later failed to 72.30 due to overfitting

    FLAGS.n_sub_frame = 108
    FLAGS.image_network_arch = "CaffeNet_dilation"
    FLAGS.image_preprocess_pad = 0
    FLAGS.num_epochs_per_decay = 2
    FLAGS.train_stage_name = "stage_all"

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    # FCN specific
    FLAGS.dropout_LSTM_keep_prob = 0.1

def car_discrete_fcn_dim_reduction_nopad_ptrain(phase):
    # should be init from car_discrete_fcn_near_dropout

    # make dir and copy the initial model
    # cd /data/yang/si/ && python scripts/train_car_stop.py train car_discrete_fcn_dim_reduction_nopad_ptrain
    # ssh -N -L 7252:localhost:7252 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    tunable_config()

    id_config("data/" + tag, 7252)

    if phase == "train":
        # for testing purpose
        set_gpu("4,5,6,7")
        FLAGS.balance_drop_prob = 0.3
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "eval":
        set_gpu("3")
        # FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
        # FLAGS.data_dir = "/y/yang/tfrecord_fix_speed"
        FLAGS.data_dir = "/data/nx-bdd-20160929/tfrecord_fix_speed"
    elif phase == "stat":
        FLAGS.data_dir = "/scratch/tfrecord_fix_speed/"
    elif phase == "board":
        set_gpu("7")

    common_discrete_settings(phase,
                             tag,
                             isFCN=False,
                             visEval=False)

    # make dir and copy the checkpoint before finetuning
    FLAGS.training_step_offset = 229000 #211910
    # stage one has reached 76.49 and platued

    FLAGS.n_sub_frame = 45 #108
    FLAGS.image_network_arch = "CaffeNet_dilation"
    FLAGS.image_preprocess_pad = 0
    FLAGS.num_epochs_per_decay = 2
    FLAGS.train_stage_name = "stage_all"

    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    # FCN specific
    FLAGS.dropout_LSTM_keep_prob = 0.1

    # ptrain related:
    FLAGS.city_data = 1
    FLAGS.segmentation_network_arch = "CaffeNet_dilation8"
    FLAGS.early_split = True
    FLAGS.cnn_split = 'drop6'
    if phase == "train":
        FLAGS.city_image_list = '/backup/BDDNexar/Harry_config/Color_train_harry.txt'
        FLAGS.city_label_list = '/backup/BDDNexar/Harry_config/TrainLabels_train_harry.txt'
    elif phase == "eval":
        FLAGS.city_image_list = '/backup/BDDNexar/Harry_config/Color_val_harry.txt'
        FLAGS.city_label_list = '/backup/BDDNexar/Harry_config/TrainLabels_val_harry.txt'



######################################################################################
############### Common setting for camera ready ##########################
######################################################################################
# discrete
def common_final_settings(phase, tag, port, basenet="32s", visEval=False, ptrain=False):
    # tunable config setting, that are not covered by common_discrete_setting
    FLAGS.arch_selection = "LRCN"
    FLAGS.ego_previous_nstep = 30
    FLAGS.max_steps = 10000000

    # id_config
    FLAGS.train_dir = "data/" + tag
    FLAGS.tensorboard_port = port

    # since we use 228*228, no need to truncate video
    common_discrete_settings(phase, tag, False, visEval)
    FLAGS.n_sub_frame = 108
    FLAGS.sleep_per_iteration = 1.0 / 3.0


    # disable balance drop since it has no effect
    FLAGS.balance_drop_prob = -1.0
    FLAGS.data_dir = "/data/yang/data/tfrecord_20170329"

    if basenet == "32s":
        FLAGS.image_network_arch = "CaffeNet"
    elif basenet == "16s":
        FLAGS.image_network_arch = "CaffeNet_dilation"
        FLAGS.image_preprocess_pad = 0
    elif basenet == "8s":
        FLAGS.image_network_arch = "CaffeNet_dilation8"
        FLAGS.image_preprocess_pad = 0

    FLAGS.resize_images = "228,228"
    FLAGS.unique_experiment_name = tag
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_dim_reduction = False
    FLAGS.add_avepool_after_dim_reduction = True

    if phase == "train":
        # ensure that the data provider is not the bottleneck
        FLAGS.num_readers = 4
        FLAGS.num_preprocess_threads = 8
        FLAGS.num_batch_join = 8
    if phase == "eval":
        FLAGS.save_best_model = True
        if visEval:
            FLAGS.pdf_normalize_bins = False
            FLAGS.use_simplifed_continuous_vis = True
            FLAGS.save_best_model = False

    if ptrain:
        FLAGS.city_data = 1
        FLAGS.segmentation_network_arch = "CaffeNet_dilation8"
        FLAGS.early_split = False
        #FLAGS.cnn_split = 'drop6'
        if phase == "train":
            FLAGS.city_image_list = '/backup/BDDNexar/Harry_config/Color_train_harry.txt'
            FLAGS.city_label_list = '/backup/BDDNexar/Harry_config/TrainLabels_train_harry.txt'
        elif phase == "eval":
            FLAGS.city_image_list = '/backup/BDDNexar/Harry_config/Color_val_harry.txt'
            FLAGS.city_label_list = '/backup/BDDNexar/Harry_config/TrainLabels_val_harry.txt'

    if phase == "test":
        FLAGS.subset="test"
        # TODO: polish to make it correct
        FLAGS.eval_method = "car_discrete"
        FLAGS.output_visualizations = False
        FLAGS.run_once = True
        FLAGS.sleep_per_iteration = 0.0
        FLAGS.city_data = 0
        FLAGS.num_preprocess_threads = 4

        # find the ".bestmodel" if possible
        best_models = []
        for f in os.listdir(FLAGS.train_dir):
            if f.endswith(".bestmodel"):
                best_models.append(f)
        if len(best_models) >= 1:
            assert(len(best_models) == 1)
            FLAGS.pretrained_model_checkpoint_path = os.path.join(FLAGS.train_dir, best_models[0])
            print("found best model", FLAGS.pretrained_model_checkpoint_path)
        else:
            print("no best model found")

    FLAGS.release_batch = True

def common_final_settings_continous(phase, tag, port, basenet="32s", visEval=False, ptrain=False):
    common_final_settings(phase, tag, port, basenet, visEval, ptrain)
    if phase == "eval" or phase == "test":
        FLAGS.eval_method = "car_continuous"
    FLAGS.sub_arch_selection = "car_loc_xy"

def camera_cnn_lstm(phase):
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py train camera_cnn_lstm
    # ssh -N -L 7253:localhost:7253 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("0")
        FLAGS.batch_size = 3 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test": #TODO finish support for testing
        set_gpu("0")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7253)

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 226053
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 388001
        FLAGS.train_stage_name = "stage_all"

def camera_tcnn9(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py train camera_tcnn9
    # ssh -N -L 7254:localhost:7254 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("1")
        FLAGS.batch_size = 2 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("0")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7254)

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 226053
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 391000
        FLAGS.train_stage_name = "stage_all"

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 9
    FLAGS.cnn_fc_hidden_units = 64

def camera_tcnn9_drop05(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py train camera_tcnn9_drop05
    # ssh -N -L 7255:localhost:7255 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("2")
        FLAGS.batch_size = 2 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("1")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7255)

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 226053
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 389000
        FLAGS.train_stage_name = "stage_all"

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 9
    FLAGS.cnn_fc_hidden_units = 64
    FLAGS.dropout_LSTM_keep_prob = 0.5

def camera_tcnn1(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py train camera_tcnn1
    # ssh -N -L 7256:localhost:7256 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("3")
        FLAGS.batch_size = 2 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("1")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7256)

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 226053
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 416000
        FLAGS.train_stage_name = "stage_all"

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64

def camera_fcn_lstm(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py eval camera_fcn_lstm
    # ssh -N -L 7257:localhost:7257 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("4")
        FLAGS.batch_size = 1 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("2")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7257,
                          basenet="8s")

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 226053
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 308000
        FLAGS.train_stage_name = "stage_all"

def camera_cnn_lstm_speed(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py eval camera_cnn_lstm_speed
    # ssh -N -L 7258:localhost:7258 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("6")
        FLAGS.batch_size = 3 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("2")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7258)

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 226053
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 372000
        FLAGS.train_stage_name = "stage_all"

    FLAGS.use_previous_speed_feature = True

def camera_speed_only(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py train camera_speed_only
    # ssh -N -L 7259:localhost:7259 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("0")
        FLAGS.batch_size = 10
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("3")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7259)

    FLAGS.num_epochs_per_decay = 8

    FLAGS.training_step_offset = 226053
    FLAGS.train_stage_name = "stage_all"

    FLAGS.use_previous_speed_feature = True
    FLAGS.use_image_feature = False
    FLAGS.no_image_input = True

    FLAGS.num_batch_join = 1
    FLAGS.num_preprocess_threads = 1

def camera_tcnn1_weight_decay(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py train camera_tcnn1_weight_decay
    # ssh -N -L 7263:localhost:7263 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("3")
        FLAGS.batch_size = 3 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("1")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7263)

    FLAGS.num_epochs_per_decay = 4
    if True:
        FLAGS.training_step_offset = 226053
        FLAGS.train_stage_name = "stage_all"

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64

def camera_tcnn1_weight_decay_stage(phase):
    # TODO start this experiment
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py train camera_tcnn1_weight_decay_stage
    # ssh -N -L 7265:localhost:7265 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("0")
        FLAGS.batch_size = 3 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("0")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7265)


    if False:
        FLAGS.training_step_offset = 226054
        FLAGS.train_stage_name = "stage_lstm"
        FLAGS.num_epochs_per_decay = 4
    else:
        FLAGS.training_step_offset = 269000
        FLAGS.train_stage_name = "stage_all"
        FLAGS.num_epochs_per_decay = 4

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64

def camera_tcnn1_weight_decay_stage_aug(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py board camera_tcnn1_weight_decay_stage_aug
    # ssh -N -L 7266:localhost:7266 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("0")
        FLAGS.batch_size = 3 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("0")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7266)

    FLAGS.training_step_offset = 284001
    FLAGS.train_stage_name = "stage_all"
    FLAGS.num_epochs_per_decay = 4

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64
    FLAGS.use_data_augmentation = True

def camera_tcnn1_weight_decay_aug_finetune(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py board camera_tcnn1_weight_decay_aug_finetune
    # ssh -N -L 7267:localhost:7267 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("0")
        FLAGS.batch_size = 3 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("0")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7267)

    FLAGS.training_step_offset = 226053
    FLAGS.train_stage_name = "stage_classic_finetune"
    FLAGS.num_epochs_per_decay = 4

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64
    FLAGS.use_data_augmentation = True

def camera_tcnn1_weight_decay_aug_stage(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py train camera_tcnn1_weight_decay_aug_stage
    # ssh -N -L 7268:localhost:7268 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("2")
        FLAGS.batch_size = 3 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("1")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7268)

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 226053
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 254000
        FLAGS.train_stage_name = "stage_all"

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64
    FLAGS.use_data_augmentation = True

def camera_tcnn1_weight_decay_aug_stage_drop6(phase):
    # TODO: run this
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py train camera_tcnn1_weight_decay_aug_stage_drop6
    # ssh -N -L 7269:localhost:7269 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("2")
        FLAGS.batch_size = 3 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("1")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7269)

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 226053
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 239000
        FLAGS.train_stage_name = "stage_all"
        #FLAGS.initial_learning_rate = 1e-5

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64
    FLAGS.use_data_augmentation = True
    FLAGS.cnn_feature = "drop7"

def camera_tcnn1_weight_decay_aug_stage_drop7_keep01(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py train camera_tcnn1_weight_decay_aug_stage_drop7_keep01
    # ssh -N -L 7270:localhost:7270 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("2")
        FLAGS.batch_size = 3 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("1")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7270)

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 226054
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 262000
        FLAGS.train_stage_name = "stage_all"

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64
    FLAGS.use_data_augmentation = True
    FLAGS.cnn_feature = "drop7"
    FLAGS.basenet_keep_prob = 0.1

def camera_tcnn1_no_bias_decay(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py train camera_tcnn1_no_bias_decay
    # ssh -N -L 7274:localhost:7274 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("3")
        FLAGS.batch_size = 3 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("0")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7274)

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 226054
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 484000
        FLAGS.train_stage_name = "stage_all"

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64

    #FLAGS.use_data_augmentation = True
    FLAGS.cnn_feature = "drop7"
    #FLAGS.basenet_keep_prob = 0.1
    FLAGS.no_batch_norm = True

    FLAGS.weight_decay_exclude_bias = True

############################experiment for continous case###################################
def camera_continous_linear(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py eval camera_continous_linear

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("0")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("7")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings_continous(phase,
                                    tag,
                                    7260,
                                    basenet = "32s",
                                    visEval = False)
    common_v2()

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_all"

    FLAGS.discretize_max_angle = math.pi / 2 * 0.99
    FLAGS.discretize_max_speed = 30 * 0.99
    FLAGS.discretize_bin_type = "linear"
    FLAGS.discretize_n_bins = 180
    FLAGS.discretize_label_gaussian_sigma = 0.5

    # change back from v3 to v2
    #FLAGS.class_balance_epsilon = 0.5
    #FLAGS.class_balance_path = "data/" + tag + "/empirical_dist"

def camera_continous_log(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py test camera_continous_log

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("1")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("2")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings_continous(phase,
                                    tag,
                                    7261)
    common_v2()

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_all"

    FLAGS.discretize_max_angle = math.pi / 2 * 0.99
    FLAGS.discretize_max_speed = 30 * 0.99
    FLAGS.discretize_bin_type = "log"
    FLAGS.discretize_n_bins = 21
    FLAGS.discretize_label_gaussian_sigma = 0.5

    FLAGS.class_balance_epsilon = 0.5
    FLAGS.class_balance_path = "data/" + tag + "/empirical_dist"

def camera_continous_adaptive(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py test camera_continous_adaptive

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("2")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("3")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings_continous(phase,
                                    tag,
                                    7262)
    common_v2()

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_all"

    FLAGS.discretize_bin_type = "custom"
    FLAGS.discretize_n_bins = 22
    FLAGS.discretize_label_gaussian_sigma = 0.5

    FLAGS.class_balance_epsilon = 0.5
    FLAGS.class_balance_path = "data/" + tag + "/empirical_dist"

def camera2_continous_linear(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py board camera2_continous_linear

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("4")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("1")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings_continous(phase,
                                    tag,
                                    7289)
    common_v2()

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_all"

    FLAGS.discretize_max_angle = math.pi / 2 * 0.99
    FLAGS.discretize_max_speed = 30 * 0.99
    FLAGS.discretize_bin_type = "linear"
    FLAGS.discretize_n_bins = 22
    FLAGS.discretize_label_gaussian_sigma = 0.5

def camera2_continous_log(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py board camera2_continous_log

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("5")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("1")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings_continous(phase,
                                    tag,
                                    7290)
    common_v2()

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_all"

    FLAGS.discretize_max_angle = math.pi / 2 * 0.99
    FLAGS.discretize_max_speed = 30 * 0.99
    FLAGS.discretize_bin_type = "log"
    FLAGS.discretize_n_bins = 21
    FLAGS.discretize_label_gaussian_sigma = 0.5

def camera2_continous_adaptive(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py board camera2_continous_adaptive

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("6")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("1")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings_continous(phase,
                                    tag,
                                    7291)
    common_v2()

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_all"

    FLAGS.discretize_bin_type = "custom"
    FLAGS.discretize_n_bins = 22
    FLAGS.discretize_label_gaussian_sigma = 0.5

def camera3_continous_log(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py train camera3_continous_log

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("1")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("6")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings_continous(phase,
                                    tag,
                                    7297)
    common_v2()

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 20000
        FLAGS.train_stage_name = "stage_all"

    FLAGS.discretize_max_angle = math.pi / 2 * 0.99
    FLAGS.discretize_min_angle = 0.1 / 180 * math.pi

    FLAGS.discretize_max_speed = 30 * 0.99
    FLAGS.discretize_min_speed = 0.1
    FLAGS.discretize_bin_type = "log"
    FLAGS.discretize_n_bins = 179
    FLAGS.discretize_label_gaussian_sigma = 0.5

def camera3_continous_datadriven(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py board camera3_continous_datadriven

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("2")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("5")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings_continous(phase,
                                    tag,
                                    7298)
    common_v2()

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_all"

    # TODO: change linear to datadriven, implement datadriven method
    FLAGS.discretize_bin_type = "datadriven"
    FLAGS.discretize_n_bins = 181
    FLAGS.discretize_label_gaussian_sigma = 0.5
    FLAGS.discretize_max_speed = 30 * 0.99

    FLAGS.discretize_datadriven_stat_path = "data/" + tag + "/empirical_dist_dataDriven.npy"

############################experiment for privilege training###################################
def ptrain_weight_decay(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py eval ptrain_weight_decay
    # ssh -N -L 7264:localhost:7264 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("1")
        FLAGS.batch_size = 1 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("0")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings(phase,
                          tag,
                          7264,
                          basenet="32s",
                          visEval=False,
                          ptrain=True)

    FLAGS.num_epochs_per_decay = 4
    FLAGS.training_step_offset = 226054
    FLAGS.train_stage_name = "stage_all"

def ptrain_weight_decay_lstm(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py board ptrain_weight_decay_lstm
    # ssh -N -L 7272:localhost:7272 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard and record the experiment on excel

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("4")
        FLAGS.batch_size = 1 * FLAGS.num_gpus
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("0")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings(phase,
                          tag,
                          7272,
                          basenet="32s",
                          visEval=False,
                          ptrain=True)

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 226053
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 276000
        FLAGS.train_stage_name = "stage_all"

    #FLAGS.use_data_augmentation = True
    FLAGS.cnn_feature = "drop7"
    #FLAGS.basenet_keep_prob = 0.1
    FLAGS.no_batch_norm = True

def ptrain_1000(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py board ptrain_1000

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("3")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("0")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings(phase,
                          tag,
                          7280,
                          basenet="32s",
                          visEval=False,
                          ptrain=True)
    common_v2()

    FLAGS.num_epochs_per_decay = 20
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 7000
        FLAGS.train_stage_name = "stage_all"

    FLAGS.retain_first_k_training_example = 1000

    if phase == "test":
        FLAGS.pretrained_model_checkpoint_path = "data/" + tag + "/model.ckpt-30001"
        # unbias test loss=0.977777, accuracy=70.7986, valid accuracy=70.08-69.81, diff=0.8

def ptrain_1000_baseline(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py board ptrain_1000_baseline

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("4")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("0")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings(phase,
                          tag,
                          7281)
    common_v2()

    FLAGS.num_epochs_per_decay = 20
    if False:
        FLAGS.training_step_offset = 1
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 9000
        FLAGS.train_stage_name = "stage_all"

    FLAGS.retain_first_k_training_example = 1000

    if phase == "test":
        # TODO
        FLAGS.pretrained_model_checkpoint_path = "data/" + tag + "/model.ckpt-30001"


def ptrain_1000_weight10_0(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py board ptrain_1000_weight10_0

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("3")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("5")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings(phase,
                          tag,
                          7282,
                          basenet="32s",
                          visEval=False,
                          ptrain=True)
    common_v2()

    FLAGS.num_epochs_per_decay = 20
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 12056
        FLAGS.train_stage_name = "stage_all"

    FLAGS.retain_first_k_training_example = 1000
    FLAGS.ptrain_weight = 10.0

def ptrain_1000_weight0_1(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py board ptrain_1000_weight0_1

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("4")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("7")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings(phase,
                          tag,
                          7283,
                          basenet="32s",
                          visEval=False,
                          ptrain=True)
    common_v2()

    FLAGS.num_epochs_per_decay = 20
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 12056
        FLAGS.train_stage_name = "stage_all"

    FLAGS.retain_first_k_training_example = 1000
    FLAGS.ptrain_weight = 0.1

def ptrain_1000_weight10_0_extra(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py test ptrain_1000_weight10_0_extra

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("5")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("0")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings(phase,
                          tag,
                          7284,
                          basenet="32s",
                          visEval=False,
                          ptrain=True)
    common_v2()

    FLAGS.num_epochs_per_decay = 20
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 12056
        FLAGS.train_stage_name = "stage_all"

    FLAGS.retain_first_k_training_example = 1000
    FLAGS.ptrain_weight = 10.0
    FLAGS.early_split = True
    FLAGS.cnn_split = 'drop7'

    if phase == "test":
        FLAGS.pretrained_model_checkpoint_path = "data/" + tag + "/model.ckpt-33001"
        # unbias test loss=0.896896, accuracy=0.699281, valid accuracy: worse than 70.09, not known exactly

def ptrain_1000_weight100_0_extra(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py test ptrain_1000_weight100_0_extra

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("6")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("7")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings(phase,
                          tag,
                          7285,
                          basenet="32s",
                          visEval=False,
                          ptrain=True)
    common_v2()

    FLAGS.num_epochs_per_decay = 20
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 12056
        FLAGS.train_stage_name = "stage_all"

    FLAGS.retain_first_k_training_example = 1000
    FLAGS.ptrain_weight = 100.0
    FLAGS.early_split = True
    FLAGS.cnn_split = 'drop7'

    if phase == "test":
        FLAGS.pretrained_model_checkpoint_path = "data/" + tag + "/model.ckpt-33001"
        # unbias test loss=0.790841, accuracy=69.8542, valid accuracy = 69.13, diff = 0.7

def ptrain_1000_FCN(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py test ptrain_1000_FCN

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("0")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("7")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings(phase,
                          tag,
                          7286,
                          basenet="8s",
                          visEval=False,
                          ptrain=True)
    common_v2()

    FLAGS.num_epochs_per_decay = 20
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 8500
        FLAGS.train_stage_name = "stage_all"

    FLAGS.retain_first_k_training_example = 1000

    if phase == "test":
        FLAGS.pretrained_model_checkpoint_path = "data/" + tag + "/model.ckpt-14001.bak"
        # unbias test loss=0.977777, accuracy=70.7986, valid accuracy=70.08-69.81, diff=0.8

def ptrain_1000_baseline_FCN(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py train ptrain_1000_baseline_FCN

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("0")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("6")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings(phase,
                          tag,
                          7287,
                          basenet="8s")
    common_v2()

    FLAGS.num_epochs_per_decay = 20
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 12000
        FLAGS.train_stage_name = "stage_all"

    FLAGS.retain_first_k_training_example = 1000

    if phase == "test":
        FLAGS.pretrained_model_checkpoint_path = "data/" + tag + "/model.ckpt-18001.bak"

def ptrain_5000_FCN(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py test ptrain_5000_FCN

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("2,3")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("7")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings(phase,
                          tag,
                          7292,
                          basenet="8s",
                          visEval=False,
                          ptrain=True)
    common_v2()

    FLAGS.num_epochs_per_decay = 20
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_all"

    FLAGS.retain_first_k_training_example = 5000

def ptrain_5000_FCN_baseline(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py test ptrain_5000_FCN_baseline

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("4,5")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("4")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings(phase,
                          tag,
                          7293,
                          basenet="8s")
    common_v2()

    FLAGS.num_epochs_per_decay = 20
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_all"

    FLAGS.retain_first_k_training_example = 5000

def ptrain_100_FCN(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py train ptrain_100_FCN

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("1")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("5")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings(phase,
                          tag,
                          7295,
                          basenet="8s",
                          visEval=False,
                          ptrain=True)
    common_v2()

    FLAGS.num_epochs_per_decay = 100
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 5000
        FLAGS.train_stage_name = "stage_all"

    FLAGS.retain_first_k_training_example = 100

def ptrain_100_FCN_baseline(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py train ptrain_100_FCN_baseline

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("2")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("6")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings(phase,
                          tag,
                          7296,
                          basenet="8s")
    common_v2()

    FLAGS.num_epochs_per_decay = 100
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 7000
        FLAGS.train_stage_name = "stage_all"

    FLAGS.retain_first_k_training_example = 100


def ptrain_1000_segpretrain(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py eval ptrain_1000_segpretrain

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("2")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("3")
    elif phase == "stat":
        pass
    elif phase == "board":
        pass

    common_final_settings(phase,
                          tag,
                          7288,
                          basenet="32s",
                          visEval=False,
                          ptrain=True)
    common_v2()

    FLAGS.num_epochs_per_decay = 20
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_all"
        if phase == "train":
            FLAGS.omit_action_loss = True
    else:
        FLAGS.city_data = 0
        FLAGS.training_step_offset = 22000
        FLAGS.train_stage_name = "stage_all"
        FLAGS.sleep_per_iteration = 0.0

    FLAGS.retain_first_k_training_example = 1000

    if phase == "test":
        FLAGS.pretrained_model_checkpoint_path = "data/" + tag + "/model.ckpt-30001"

########################################rerun all experiments based on new base arch####################################
def common_v2():
    # FLAGS.use_data_augmentation = True
    FLAGS.cnn_feature = "drop7"
    # FLAGS.basenet_keep_prob = 0.1
    FLAGS.no_batch_norm = True
    FLAGS.weight_decay_exclude_bias = False

def camera2_tcnn9(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py test camera2_tcnn9

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("0")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("0")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7275)
    common_v2()

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 103850
        FLAGS.train_stage_name = "stage_all"

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 9
    FLAGS.cnn_fc_hidden_units = 64

def camera2_tcnn3(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py test camera2_tcnn3

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("0")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("7")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7294)
    common_v2()

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 21000
        FLAGS.train_stage_name = "stage_all"

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 3
    FLAGS.cnn_fc_hidden_units = 64

def camera2_tcnn9_drop05(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py eval camera2_tcnn9_drop05
    # ssh -N -L 7276:localhost:7276 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("1")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("1")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7276)
    common_v2()

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 84000
        FLAGS.train_stage_name = "stage_all"

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 9
    FLAGS.cnn_fc_hidden_units = 64
    FLAGS.dropout_LSTM_keep_prob = 0.5

def camera2_fcn_lstm(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py test camera2_fcn_lstm
    # ssh -N -L 7277:localhost:7277 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("2")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("6")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7277,
                          basenet="8s")
    common_v2()

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 34000
        FLAGS.train_stage_name = "stage_all"

def camera2_cnn_speed(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py test camera2_cnn_speed

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("1")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("1")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7278)
    common_v2()

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 82675
        FLAGS.train_stage_name = "stage_all"

    FLAGS.use_previous_speed_feature = True

def camera2_speed_only(phase):
    # record the experiment on excel
    # make dir and copy the initial model from car_discrete_cnn_224_224
    # cd /data/yang/si/ && python scripts/train_car_stop.py test camera2_speed_only
    # ssh -N -L 7279:localhost:7279 leviathan.ist.berkeley.edu &
    # open the browser for tensorboard

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("1")
        FLAGS.batch_size = 10
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("1")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7279)
    common_v2()

    FLAGS.num_epochs_per_decay = 8

    FLAGS.training_step_offset = 0
    FLAGS.train_stage_name = "stage_all"

    FLAGS.use_previous_speed_feature = True
    FLAGS.use_image_feature = False
    FLAGS.no_image_input = True

    FLAGS.num_batch_join = 1
    FLAGS.num_preprocess_threads = 1

# CNN lstm baseline
def camera_cnn_lstm_no_batch_norm(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py test camera_cnn_lstm_no_batch_norm

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("3")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("1")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7273)
    common_v2()

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 19433
        FLAGS.train_stage_name = "stage_all"

# TCNN1 baseline
def camera_tcnn1_no_batch_norm(phase):
    # record the experiment on excel
    # cd /data/yang/si/ && python scripts/train_car_stop.py test camera_tcnn1_no_batch_norm

    tag = inspect.stack()[0][3]
    if phase == "train":
        set_gpu("2")
    elif phase == "eval" or phase == "test":  # TODO finish support for testing
        set_gpu("5")
    elif phase == "stat":
        pass
    elif phase == "board":
        set_gpu("0")

    common_final_settings(phase,
                          tag,
                          7271)
    common_v2()

    FLAGS.num_epochs_per_decay = 4
    if False:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = 24732
        FLAGS.train_stage_name = "stage_all"

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64

if __name__ == '__main__':
    # python script_name train/eval/board small_config
    # cd /data/hxu/new_scale/scale/tfcnn/ & python scripts/train_car_stop.py eval car_discrete_cnn_low_small
    # cd /data/hxu/new_scale/scale/tfcnn/ & python scripts/train_car_stop.py train car_discrete_cnn_low_small
    phase=sys.argv[1]
    small_config=sys.argv[2]

    common_config(phase)
    globals()[small_config](phase)
    common_config_post(phase)

    work = { "train": train,
             "eval" : eval,
             "test": test,
             "board": tensorboard,
             "stat": stat}[phase]
    work()
