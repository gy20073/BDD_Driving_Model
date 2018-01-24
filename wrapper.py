'''
This is a wrapper for easier evaluation.

The original evaluation code is mainly written for evaluation on a validation set. It requires the dataset be processed
into the designated TFRecord format. This wrapper aims to be a lightweight interface without requiring the TFRecord
format input. Instead, it accepts inputs of images, and output actions on the fly.
'''

import tensorflow as tf
import models.car_stop_model as model
from scipy.misc import imresize
import numpy as np

# The following import populates some FLAGS default value
import data_providers.nexar_large_speed
import batching
import dataset
import util_car

FLAGS = tf.app.flags.FLAGS
flags_passthrough = FLAGS._parse_flags()
from config import common_config, common_config_post
import importlib, sys, time

IMSZ = 228

class Wrapper:
    def __init__(self, model_config_name, model_path, truncate_len=20, config_name="config", config_path=".", is_lstm=False, post_config_func=None):
        self.is_lstm = is_lstm
        if is_lstm:
            assert truncate_len==1, \
                "using lstm should set truncate_len to 1, otherwise waste of computing resource"

        # currently, we use a sliding window fashion for evaluation, that's inefficient but convenient to implement
        self.truncate_len = truncate_len
        self.latest_frames = []
        for _ in range(truncate_len):
            self.latest_frames.append(np.zeros((IMSZ, IMSZ, 3), dtype=np.uint8))

        # call the config.py for setup
        sys.path.append(config_path)
        config = importlib.import_module(config_name)
        common_config("eval")
        config_fun = getattr(config, model_config_name)
        config_fun("eval")
        common_config_post("eval")
        if post_config_func is not None:
            config_fun_post = getattr(config, post_config_func)
            config_fun_post()

        # Tensors in has the format: [images, speed] for basic usage, excluding only_seg
        # For now, we decide not to support previous speed as input, thus we use a fake speed (-1) now
        # and ensures the speed is not used by asserting FLAGS.use_previous_speed_feature==False
        assert (not hasattr(FLAGS, "use_previous_speed_feature")) or (FLAGS.use_previous_speed_feature == False)
        # batch size 1 all the time, length undetermined, width and height are IMSZ
        self.tensors_in = tf.placeholder(tf.uint8, shape=(1, truncate_len, IMSZ, IMSZ, 3), name="images_input")
        self.speed = None

        if is_lstm:
            self.initial_state = ((tf.placeholder(tf.float32,
                                                  shape=(1, int(FLAGS.lstm_hidden_units)),
                                                  name="state_placeholder1"),
                                  tf.placeholder(tf.float32,
                                                 shape=(1, int(FLAGS.lstm_hidden_units)),
                                                 name="state_placeholder2")), )

            FLAGS.phase = "rnn_inference"
        else:
            self.initial_state = None
        logits_all = model.inference([self.tensors_in, self.speed], -1, for_training=False,
                                     initial_state=self.initial_state)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        config = tf.ConfigProto(intra_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        saver.restore(self.sess, model_path)

        self.logits = logits_all[0]
        if is_lstm:
            self.state_tensor = logits_all[-1]
            self.state_value=[[np.zeros((1, int(FLAGS.lstm_hidden_units)), dtype=np.float32),
                               np.zeros((1, int(FLAGS.lstm_hidden_units)), dtype=np.float32)]]

        init_op = tf.initialize_local_variables()
        self.sess.run(init_op)

    def observe_a_frame(self, image):
        '''
        Assuming the input frequency is 3Hz

        Args:
            image: a 3D numpy array of size H*W*3

        Returns:
            an action output from the model
        '''
        image = self.process_frame(image)
        self.latest_frames.append(image)
        if len(self.latest_frames) > self.truncate_len:
            self.latest_frames = self.latest_frames[-self.truncate_len:]

        batch = np.stack(self.latest_frames, axis=0)
        batch = batch[np.newaxis]

        time0 = time.time()
        fd = {self.tensors_in: batch}
        if self.is_lstm:
            fd[self.initial_state[0][0]] = self.state_value[0][0]
            fd[self.initial_state[0][1]] = self.state_value[0][1]
            logits_v, self.state_value = self.sess.run([self.logits, self.state_tensor], feed_dict=fd)
        else:
            logits_v = self.sess.run(self.logits, feed_dict=fd)

        print("only forward pass", time.time()-time0)

        logits_v = logits_v[-1:, :]
        # discrete output method
        '''
        # meaning for discrete actions
        turn_str2int = {'not_sure': -1, 'straight': 0, 'slow_or_stop': 1,
                        'turn_left': 2, 'turn_right': 3,
                        'turn_left_slight': 4, 'turn_right_slight': 5,}
        '''
        # continuous output method
        # MAPs = model.continous_MAP([logits_all])
        return [logits_v]

    def process_frame(self, image):
        return imresize(image, (IMSZ, IMSZ))

    def continuous_muti_querys_pdf(self, logits, querys):
        return model.continous_pdf(logits, querys, "multi_querys")

    def continuous_MAP(self, logits):
        return model.continous_MAP(logits)

    def generate_visualization(self, image, logits, method="vis_continuous"):
        return util_car.continuous_vis_single_image(image, logits, method)
