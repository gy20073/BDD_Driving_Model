import math
import sys
import os
from subprocess import call
import inspect
sys.path.append('../')

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

############################Set those path before use###################################
FLAGS.pretrained_model_path = "/data/yang/si/data/pretrained_models/tf.caffenet.bin"
FLAGS.data_dir = "/data/yang_cache/tfrecord_release/tfrecords"

# for privilege training: segmentation image index and labels
train_city_image_list = '/backup/BDDNexar/Harry_config/Color_train_harry.txt'
train_city_label_list = '/backup/BDDNexar/Harry_config/TrainLabels_train_harry.txt'
eval_city_image_list = '/backup/BDDNexar/Harry_config/Color_val_harry.txt'
eval_city_label_list = '/backup/BDDNexar/Harry_config/TrainLabels_val_harry.txt'

############################discrete action###################################
def discrete_speed_only(phase):
    tag = inspect.stack()[0][3]
    set_gpu_ids(phase, "0", "0")
    if phase == "train":
        FLAGS.batch_size = 10

    common_final_settings(phase,
                          tag,
                          7279)

    FLAGS.num_epochs_per_decay = 8

    FLAGS.train_stage_name = "stage_all"

    FLAGS.use_previous_speed_feature = True
    FLAGS.use_image_feature = False
    FLAGS.no_image_input = True

    FLAGS.num_batch_join = 1
    FLAGS.num_preprocess_threads = 1

def discrete_tcnn1(phase):
    tag = inspect.stack()[0][3]
    set_gpu_ids(phase, "2", "0")
    common_final_settings(phase,
                          tag,
                          7271)
    set_train_stage(False, 24732)

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 1
    FLAGS.cnn_fc_hidden_units = 64

def discrete_tcnn3(phase):
    tag = inspect.stack()[0][3]
    set_gpu_ids(phase, "3", "4")
    common_final_settings(phase,
                          tag,
                          7294)
    set_train_stage(False, 21000)

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 3
    FLAGS.cnn_fc_hidden_units = 64

def discrete_tcnn9(phase):
    tag = inspect.stack()[0][3]
    set_gpu_ids(phase, "4", "5")
    common_final_settings(phase,
                          tag,
                          7275)
    set_train_stage(False, 103850)

    # specific setting
    FLAGS.arch_selection = "CNN_FC"
    FLAGS.history_window = 9
    FLAGS.cnn_fc_hidden_units = 64

def discrete_cnn_lstm(phase):
    tag = inspect.stack()[0][3]
    set_gpu_ids(phase, "5", "6")
    common_final_settings(phase,
                          tag,
                          7273)
    set_train_stage(False, 19433)

def discrete_cnn_lstm_speed(phase):
    tag = inspect.stack()[0][3]
    set_gpu_ids(phase, "6", "7")
    common_final_settings(phase,
                          tag,
                          7278)
    set_train_stage(False, 82675)

    FLAGS.use_previous_speed_feature = True

def discrete_fcn_lstm(phase):
    tag = inspect.stack()[0][3]
    set_gpu_ids(phase, "7", "3")
    common_final_settings(phase,
                          tag,
                          7277,
                          basenet="8s")
    set_train_stage(False, 34000)
    FLAGS.num_epochs_per_decay = 12


############################continuous action###################################
def continuous_linear_bin(phase):
    tag = inspect.stack()[0][3]
    set_gpu_ids(phase, "0", "5")
    common_final_settings_continous(phase,
                                    tag,
                                    7260)
    set_train_stage(False, 110001)

    FLAGS.discretize_max_angle = math.pi / 2 * 0.99
    FLAGS.discretize_max_speed = 30 * 0.99
    FLAGS.discretize_label_gaussian_sigma = 0.5

    FLAGS.discretize_bin_type = "linear"
    FLAGS.discretize_n_bins = 180


def continuous_log_bin(phase):
    tag = inspect.stack()[0][3]
    set_gpu_ids(phase, "1", "6")
    common_final_settings_continous(phase,
                                    tag,
                                    7297)
    set_train_stage(False, 149001)

    FLAGS.discretize_max_angle = math.pi / 2 * 0.99
    FLAGS.discretize_min_angle = 0.1 / 180 * math.pi
    FLAGS.discretize_max_speed = 30 * 0.99
    FLAGS.discretize_min_speed = 0.1
    FLAGS.discretize_label_gaussian_sigma = 0.5

    FLAGS.discretize_bin_type = "log"
    FLAGS.discretize_n_bins = 179


def continuous_datadriven_bin(phase):
    tag = inspect.stack()[0][3]
    set_gpu_ids(phase, "2", "5")
    common_final_settings_continous(phase,
                                    tag,
                                    7298)
    set_train_stage(False, 91001)

    FLAGS.discretize_max_speed = 30 * 0.99
    FLAGS.discretize_label_gaussian_sigma = 0.5

    FLAGS.discretize_bin_type = "datadriven"
    FLAGS.discretize_n_bins = 181
    FLAGS.discretize_datadriven_stat_path = "data/" + tag + "/empirical_dist_dataDriven.npy"

    FLAGS.stat_datadriven_only = True

####################### priviledge training  #######################
def ptrain_1000_FCN(phase):
    tag = inspect.stack()[0][3]
    set_gpu_ids(phase, "3", "6")
    common_final_settings(phase,
                          tag,
                          7286,
                          basenet="8s",
                          visEval=False,
                          ptrain=True)

    FLAGS.num_epochs_per_decay = 20
    set_train_stage(False, 53001)

    FLAGS.retain_first_k_training_example = 1000

def ptrain_1000_baseline_FCN(phase):
    tag = inspect.stack()[0][3]
    set_gpu_ids(phase, "4", "5")
    common_final_settings(phase,
                          tag,
                          7287,
                          basenet="8s")

    FLAGS.num_epochs_per_decay = 20
    set_train_stage(False, 80001)

    FLAGS.retain_first_k_training_example = 1000



######################################################################################
############### shared settings ##########################
######################################################################################
def set_gpu(gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    num_gpus = len(gpus.split(","))
    FLAGS.num_gpus = num_gpus

def set_gpu_ids(phase, train, eval_or_test):
    if phase == "train":
        set_gpu(train)
    elif phase == "eval" or phase == "test":
        set_gpu(eval_or_test)

def set_train_stage(isFirstStage, offset):
    if isFirstStage:
        FLAGS.training_step_offset = 0
        FLAGS.train_stage_name = "stage_lstm"
    else:
        FLAGS.training_step_offset = offset
        FLAGS.train_stage_name = "stage_all"

# discrete
def common_final_settings(phase, tag, port, basenet="32s", visEval=False, ptrain=False):
    # resource related
    FLAGS.unique_experiment_name = tag
    FLAGS.train_dir = "data/" + tag
    FLAGS.tensorboard_port = port

    # optimization related
    FLAGS.max_steps = 10000000
    FLAGS.train_stage_name = 'stage_all'
    FLAGS.clip_gradient_threshold = 10.0
    FLAGS.momentum = 0.99
    FLAGS.num_epochs_per_decay = 4
    FLAGS.initial_learning_rate = 1e-4
    FLAGS.learning_rate_decay_factor = 0.5

    # NN architecture related
    FLAGS.arch_selection = "LRCN"
    FLAGS.sub_arch_selection = "car_discrete"
    FLAGS.lstm_hidden_units = "64"
    FLAGS.add_dropout_layer = False
    FLAGS.cnn_feature = "drop7"
    FLAGS.no_batch_norm = True
    FLAGS.weight_decay_exclude_bias = False
    FLAGS.enable_basenet_dropout = True
    FLAGS.add_dim_reduction = False
    FLAGS.add_avepool_after_dim_reduction = True

    # data related
    FLAGS.ego_previous_nstep = 30
    FLAGS.n_sub_frame = 108
    FLAGS.release_batch = True
    FLAGS.resize_images = "228,228"
    FLAGS.balance_drop_prob = -1.0

    FLAGS.decode_downsample_factor = 1
    FLAGS.temporal_downsample_factor = 5
    FLAGS.data_provider = "nexar_large_speed"
    # ground truth maker
    FLAGS.speed_limit_as_stop = 2.0
    FLAGS.stop_future_frames = 1
    FLAGS.deceleration_thres = 1
    FLAGS.no_slight_turn = True

    # conditional setup
    if basenet == "32s":
        FLAGS.image_network_arch = "CaffeNet"
    elif basenet == "16s":
        FLAGS.image_network_arch = "CaffeNet_dilation"
        FLAGS.image_preprocess_pad = 0
    elif basenet == "8s":
        FLAGS.image_network_arch = "CaffeNet_dilation8"
        FLAGS.image_preprocess_pad = 0

    if ptrain:
        FLAGS.city_data = 1
        FLAGS.segmentation_network_arch = "CaffeNet_dilation8"
        FLAGS.early_split = False
        if phase == "train":
            FLAGS.city_image_list = train_city_image_list
            FLAGS.city_label_list = train_city_label_list
        elif phase == "eval":
            FLAGS.city_image_list = eval_city_image_list
            FLAGS.city_label_list = eval_city_label_list

    if phase == "train":
        # ensure that the data provider is not the bottleneck
        FLAGS.num_readers = 4
        FLAGS.num_preprocess_threads = 8
        FLAGS.num_batch_join = 8
    elif phase == "eval":
        FLAGS.eval_method = "car_discrete"

        if visEval:
            FLAGS.output_visualizations = True
            FLAGS.run_once = True
            FLAGS.save_best_model = False

            FLAGS.subsample_factor = 10
            FLAGS.pdf_normalize_bins = False
            FLAGS.use_simplifed_continuous_vis = True
        else:
            FLAGS.output_visualizations = False
            FLAGS.run_once = False
            FLAGS.save_best_model = True

            FLAGS.eval_interval_secs = 1
            FLAGS.sleep_per_iteration = 1.0 / 4

    elif phase == "stat":
        set_gpu("0")
        FLAGS.subset = "train"

        FLAGS.stat_output_path = "data/" + tag + "/empirical_dist"
        FLAGS.eval_method = "stat_labels"
        FLAGS.no_image_input = True
        FLAGS.subsample_factor = 10
    elif phase == "board":
        set_gpu("0")
    elif phase == "test":
        FLAGS.subset="test"
        FLAGS.eval_method = "car_discrete"
        FLAGS.run_once = True
        FLAGS.city_data = 0

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

    if not (phase == "board" or phase == "stat"):
        FLAGS.batch_size = 1 * FLAGS.num_gpus


def common_final_settings_continous(phase, tag, port, basenet="32s", visEval=False, ptrain=False):
    common_final_settings(phase, tag, port, basenet, visEval, ptrain)
    if phase == "eval" or phase == "test":
        FLAGS.eval_method = "car_continuous"
    FLAGS.sub_arch_selection = "car_loc_xy"

######################################################################################
############### end of common settings ##########################
######################################################################################

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
    FLAGS.optimizer = "sgd"
    FLAGS.profile = False
    FLAGS.model_definition = "car_stop_model"
    FLAGS.num_readers = 2
    FLAGS.pretrained_model_checkpoint_path = ""
    FLAGS.num_preprocess_threads = 4
    FLAGS.display_loss = 10
    FLAGS.display_summary = 100
    FLAGS.checkpoint_interval = 5000
    FLAGS.input_queue_memory_factor = 8
    FLAGS.examples_per_shard=1
    FLAGS.use_MIMO_inputs_pipeline=True

    # related to evaluation
    FLAGS.subsample_factor=1

def common_config_post(phase):
    FLAGS.eval_dir = os.path.join(FLAGS.train_dir, "eval")
    FLAGS.checkpoint_dir = FLAGS.train_dir

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

def flags_to_cmd():
    # dict of flags to values
    d = FLAGS.__dict__["__flags"]
    out=[]
    for k, v in d.iteritems():
        print(k, v)
        out.append("--"+k+"="+str(v))
    return out

if __name__ == '__main__':
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
