import math
import sys
import os
from subprocess import call
import inspect
sys.path.append('../')

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

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


####################### priviledge training  #######################
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




######################################################################################
############### Common setting for camera ready ##########################
######################################################################################
def set_gpu(gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    num_gpus = len(gpus.split(","))
    FLAGS.num_gpus = num_gpus

def common_discrete_settings(phase, tag, isFCN, visEval):
    if phase == "train":
        FLAGS.n_sub_frame = 45 if isFCN else 108
    elif phase == "eval":
        FLAGS.balance_drop_prob = -1.0

        FLAGS.n_sub_frame = 108

        FLAGS.eval_method = "car_discrete"

        if visEval:
            FLAGS.output_visualizations = True
            FLAGS.subsample_factor = 10

            FLAGS.run_once = True
        else:
            FLAGS.output_visualizations = False
            FLAGS.eval_interval_secs = 1
            FLAGS.run_once = False
    elif phase == "stat":
        set_gpu("0")
        FLAGS.subset = "train"
        FLAGS.n_sub_frame = 108

        FLAGS.stat_output_path = "data/" + tag + "/empirical_dist"
        FLAGS.eval_method = "stat_labels"
        FLAGS.no_image_input = True
        FLAGS.subsample_factor = 10

    if not (phase == "board" or phase == "stat"):
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
    FLAGS.stop_future_frames = 1  # make sure this make sense
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

def common_v2():
    # FLAGS.use_data_augmentation = True
    FLAGS.cnn_feature = "drop7"
    # FLAGS.basenet_keep_prob = 0.1
    FLAGS.no_batch_norm = True
    FLAGS.weight_decay_exclude_bias = False

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
