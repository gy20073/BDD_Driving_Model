from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset import Dataset
import tensorflow as tf
import numpy as np
import os, random
import util
import util_car
import scipy.misc as misc
import glob
import multiprocessing
from cStringIO import StringIO
from PIL import Image
import cv2
import ctypes
import math, copy
from scipy import interpolate

FLAGS = tf.app.flags.FLAGS


ctx_channel = 19
ctx_height = 60
ctx_width = 90
len_seg_ctx = 100
#FRAMES_IN_SEG=570

city_im_channel = 3
city_seg_channel = 1
city_frames = 5
city_lock = multiprocessing.Lock()

# Visual preprocessing FLAGS
tf.app.flags.DEFINE_integer('IM_WIDTH', 640, '''width of image in TFRecord''')
tf.app.flags.DEFINE_integer('IM_HEIGHT', 360, '''height of image in TFRecord''')
tf.app.flags.DEFINE_string('resize_images', "",
                           """If not empty resize the images to the required shape. Shape format: height,width""")
tf.app.flags.DEFINE_integer('decode_downsample_factor', 1,
                            '''The original high res video is 640*360. For training purpose, this resolution '''
                            '''might be too high. This param downsample the image during jpeg decode process''')
tf.app.flags.DEFINE_integer('temporal_downsample_factor', 5,
                            '''The original video is in 15 FPS, this flag optionally downsample the video temporally'''
                            '''All other frame related operations are carried out after temporal downsampling''')

# data usage FLAGS
tf.app.flags.DEFINE_integer('n_sub_frame', 108,
                            'How many frames does one instance contains. Reduce this for larger network if memory not enough')
tf.app.flags.DEFINE_integer('stop_future_frames', 2,
                            '''Shift the stop labels * frames forward, to predict the future''')
tf.app.flags.DEFINE_float('speed_limit_as_stop', 0.3,
                            '''if the speed is less than this, then it's considered to be stopping''')
tf.app.flags.DEFINE_boolean('no_image_input', False,
                            '''if true, then don't parse the image input''')
tf.app.flags.DEFINE_float('balance_drop_prob', -1.0,
                          '''drop no stop seq with specified probability. Don't drop if it < 0''')
tf.app.flags.DEFINE_float('acceleration_thres', -1.0,
                          '''acceleration threshold, minus value for not using it''')
tf.app.flags.DEFINE_float('deceleration_thres', -1.0,
                          '''deceleration threshold, minus value for not using it''')
tf.app.flags.DEFINE_boolean('no_slight_turn', True,
                            '''if true, then don't use slight left or right turn''')
tf.app.flags.DEFINE_boolean('non_random_temporal_downsample', False,
                            '''if true, use fixed downsample method''')
tf.app.flags.DEFINE_string('fast_jpeg_decode', "default",
                            '''which type of jpeg decode to use: default, tf, pyfunc'''
                            '''tf is the fastest and could reduce CPU usage a lot, but require compilation''')
#dataset specified FLAGS
tf.app.flags.DEFINE_string('city_image_list','/data/hxu/fineGT/trainval_images.txt',
                           'the privilege training segmentation image index')
tf.app.flags.DEFINE_string('city_label_list','/data/hxu/fineGT/trainval_labels.txt', 
                           'the privilege training segmentation label image index')
tf.app.flags.DEFINE_integer('city_data',0,'if we want side training with segmentation, set this to 1')
tf.app.flags.DEFINE_integer('only_seg',0, 'if we want to only use segmentation, set this to 1')
tf.app.flags.DEFINE_integer('FRAMES_IN_SEG',540,
                            'How many frames in a single TFRecord, this is conservatively set to 36second * 15fps')
tf.app.flags.DEFINE_integer('city_batch', 1, "segmentation batch size")
tf.app.flags.DEFINE_boolean('is_small_side_info_dataset', False,
                            '''if true, use smaller segmentation dataset''')
tf.app.flags.DEFINE_boolean('low_res', False,
                            '''The BDD video dataset comes with 2 resolution, this flags optionally allow
                            to use the lower resolution dataset''')
tf.app.flags.DEFINE_float('frame_rate', 15.0, 'the frame_rate we have for the videos')
tf.app.flags.DEFINE_string('train_filename', 'train_small.txt',
                           '''the file name list we used to train. This is useful for super large dataset, such as
                           the low resolution part of BDD video dataset. It avoids to Glob all files when training start''')
tf.app.flags.DEFINE_boolean('release_batch', False, 'True if the dataset is the finally released version' )
tf.app.flags.DEFINE_boolean('use_data_augmentation', False, 'whether to augment the training data' )

tf.app.flags.DEFINE_integer('retain_first_k_training_example', -1,
                            'retrain the first k training example if >0 ')

tf.app.flags.DEFINE_boolean('use_speed_yaw', False,
                            'By default, the TFRecord gives a set of speeds in x, y coordinates. If True, we instead '
                            'directly stores the speed and yaw rate in the TFRecord. ')
tf.app.flags.DEFINE_boolean('is_MKZ_dataset', False,
                            'The newly collected data on the Lincoln MKZ')
tf.app.flags.DEFINE_string('custom_dataset_name', '',
                           'a customized dataset name')

tf.app.flags.DEFINE_boolean('use_non_random_shuffle', False,
                            'Whether to shuffle the files')

tf.app.flags.DEFINE_integer('crop_car_hood', -1,
                            'crop the bottom x pixels, if >0. Pixels counted after resize. ')
tf.app.flags.DEFINE_boolean('use_perspective_augmentation', False,
                            'Whether to perspective augmentation')

tf.app.flags.DEFINE_integer('inflate_MKZ_factor', -1,
                            'inflate the MKZ data if > 0 ')


# the newly designed class has to have those methods
# especially the reader() that reads the binary record and the
# parse_example_proto that parse a single record into an instance
class MyDataset(Dataset):
    def __init__(self, subset):
        super(MyDataset, self).__init__('nexar', subset)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        # TODO: not useful
        return 2

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data set."""
        # TODO: Edit number

        if self.subset == 'train':
            if FLAGS.retain_first_k_training_example > 0:
                return FLAGS.retain_first_k_training_example

            if FLAGS.custom_dataset_name == "nexar_MKZ":
                return 21244
            elif FLAGS.custom_dataset_name == "day_night":
                return 4977

            if FLAGS.is_MKZ_dataset:
                print("using the MKZ dataset")
                return 40

            if FLAGS.low_res:
                # TODO: change to these names
                if FLAGS.train_filename == 'train_small.txt':
                    return 25497
                elif FLAGS.train_filename == 'train_medium.txt':
                    return 192514
                elif FLAGS.train_filename == 'train_large.txt':
                    return 2000000
            elif FLAGS.is_small_side_info_dataset:
                return 945
            elif FLAGS.release_batch:
                # finalized version
                return 21204
                #return 21808
            else:
                return 28738
        if self.subset == 'validation':
            if FLAGS.custom_dataset_name == "nexar_MKZ":
                return 29
            elif FLAGS.custom_dataset_name == "nexar_vehicle_ahead":
                return 20
            elif FLAGS.custom_dataset_name == "day_night":
                return 6478

            if FLAGS.is_MKZ_dataset:
                print("using the MKZ dataset")
                return 9

            if FLAGS.low_res:
                return 12877
            elif FLAGS.is_small_side_info_dataset:
                return 101
            elif FLAGS.release_batch:
                # finalized version
                return 1867
                #return 1470
            else:
                return 1906

        if self.subset == "test":
            if FLAGS.release_batch:
                return 3561

    def download_message(self):
        print('Failed to find any nexar %s files' % self.subset)

    def data_files(self):
        """Returns a python list of all (sharded) data subset files.

        Returns:
          python list of all (sharded) data set files.
        Raises:
          ValueError: if there are not data_files matching the subset.
        """
        print(FLAGS.data_dir,'IN DATA_FILES')
        # TODO: make sure file name pattern matches
        if FLAGS.low_res:
            if self.subset == "train":
                filename = FLAGS.train_filename
            else:
                filename = 'validation_name.txt'
            filename_dir = os.path.join(FLAGS.data_dir, filename)
            data_files = []
            with open(filename_dir) as f:
                for item in f:
                    item = item.strip()
                    data_files.append(item)

        else:
            if self.subset == "train":
                pattern = "train/*.tfrecords"
            elif self.subset == 'validation':
                pattern = "validation/*.tfrecords"
            elif self.subset == "test":
                pattern = "test/*.tfrecords"
            else:
                raise ValueError("invalid dataset subset")

            tf_record_pattern = os.path.join(FLAGS.data_dir, pattern)
            data_files = tf.gfile.Glob(tf_record_pattern)
            data_files = sorted(data_files)
            if not data_files:
                print('No files found for dataset %s/%s at %s' % (self.name,
                                                                  self.subset,
                                                                  FLAGS.data_dir))
                self.download_message()
                exit(-1)

        if FLAGS.inflate_MKZ_factor > 0 and self.subset == "train":
            # manually inflate the MKZ examples
            MKZ_names = [name for name in data_files if len(name.split("/")[-1]) < 16]
            print(MKZ_names)
            data_files = data_files + MKZ_names * (FLAGS.inflate_MKZ_factor - 1)
            random.seed(a=1)
            random.shuffle(data_files)
            print("using the inflate factor")


        print('Glob Files Done...')
        if FLAGS.retain_first_k_training_example > 0 and self.subset == "train":
            data_files = sorted(data_files)
            data_files = data_files[:FLAGS.retain_first_k_training_example]

        if FLAGS.use_non_random_shuffle:
            random.seed(a=1)
            random.shuffle(data_files)
        return data_files

    def decode_jpeg(self, image_buffer, scope=None):
        if FLAGS.fast_jpeg_decode == "pyfunc":
            print("using ctypes jpeg decode...")
            lib_jpeg = ctypes.cdll.LoadLibrary('./data_providers/decode_jpeg_memory/decode_memory.so')
            global ctypes_jpeg
            ctypes_jpeg = lib_jpeg.decode_jpeg_memory_turbo
            return self.decode_jpeg_python(image_buffer, scope)
        elif FLAGS.fast_jpeg_decode=="tf":
            print("using tensorflow binary libjpeg turbo")
            decode_jpeg_batch = tf.load_op_library(
                './data_providers/decode_jpeg_memory/decode_jpeg_batch.so').decode_jpeg_batch
            assert( FLAGS.decode_downsample_factor == 1 )
            ans = decode_jpeg_batch(image_buffer, FLAGS.IM_HEIGHT, FLAGS.IM_WIDTH)
            ans.set_shape([FLAGS.FRAMES_IN_SEG // FLAGS.temporal_downsample_factor,
                           FLAGS.IM_HEIGHT, FLAGS.IM_WIDTH, 3])
            return ans
        else:
            return self.decode_jpeg_original(image_buffer, scope)

    def decode_jpeg_original(self, image_buffer, scope=None):
        with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
            # decode jpeg
            fn = lambda encoded: tf.image.decode_jpeg(encoded,
                                                      channels=3,
                                                      ratio=FLAGS.decode_downsample_factor)
            # TODO: change parallel iterations
            decoded = tf.map_fn(fn,
                                image_buffer,
                                dtype=tf.uint8,
                                parallel_iterations=1,
                                back_prop=False,
                                swap_memory=False,
                                infer_shape=True,
                                name="map_decode_jpeg")

            # move the float convertion to GPU, to save bandwidth
            # to float, to the range 0.0 - 255.0
            #images = tf.cast(decoded, dtype=tf.float32, name="image_to_float")
            decoded.set_shape([FLAGS.FRAMES_IN_SEG // FLAGS.temporal_downsample_factor,
                              FLAGS.IM_HEIGHT / FLAGS.decode_downsample_factor,
                              FLAGS.IM_WIDTH / FLAGS.decode_downsample_factor, 3])
            return decoded

    def decode_jpeg_concat(self, image_buffer, scope=None):
        with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
            length = image_buffer.get_shape()[0].value
            all = []
            for i in range(length):
                decoded = tf.image.decode_jpeg(image_buffer[i],
                                     channels=3,
                                     ratio=FLAGS.decode_downsample_factor)
                all.append(tf.expand_dims(decoded, 0))
            images = tf.concat(0, all)
            images.set_shape([FLAGS.FRAMES_IN_SEG // FLAGS.temporal_downsample_factor,
                               FLAGS.IM_HEIGHT / FLAGS.decode_downsample_factor,
                               FLAGS.IM_WIDTH / FLAGS.decode_downsample_factor, 3])
            return images

    def decode_jpeg_batch(self, image_buffer, scope=None):
        with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
            length = image_buffer.get_shape()[0].value
            queue=tf.train.string_input_producer(
                image_buffer,
                num_epochs=None,
                shuffle=False,
                capacity=2)
            image = queue.dequeue()
            decoded = tf.image.decode_jpeg(image,
                                           channels=3,
                                           ratio=FLAGS.decode_downsample_factor)
            batch = tf.train.batch([decoded],
                           batch_size=length,
                           num_threads=1,
                           capacity=length,
                           enqueue_many=False,
                           shapes=[[FLAGS.IM_HEIGHT / FLAGS.decode_downsample_factor,
                               FLAGS.IM_WIDTH / FLAGS.decode_downsample_factor, 3]],
                           dynamic_pad=False,
                           allow_smaller_final_batch=False)
            return batch

    @staticmethod
    def decode_batch(image_strs, H, W, C, downsample):
        assert (downsample == 1)
        ans = np.zeros([len(image_strs), H, W, C], dtype=np.uint8)
        for i, st in enumerate(image_strs):
            file_bytes = np.asarray(bytearray(st), dtype=np.uint8)
            t = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            ans[i, :, :, :] = t
        ans = ans[:, :, :, [2, 1, 0]]
        return ans

    @staticmethod
    def decode_batch_libturbo(image_strs, H, W, C, downsample):
        assert (downsample == 1)
        ans = np.zeros([len(image_strs), H, W, C], dtype=np.uint8)

        p3 = ctypes.c_void_p(ans.ctypes.data)
        p4 = ctypes.c_int(H)
        p5 = ctypes.c_int(W)
        print("in libturbo")
        for i, st in enumerate(image_strs):
            file_bytes = np.asarray(bytearray(st), dtype=np.uint8)
            p1 = ctypes.c_void_p(file_bytes.ctypes.data)
            p2 = ctypes.c_int(len(file_bytes))
            p6 = ctypes.c_int(H * W * C * i)
            ctypes_jpeg(p1, p2, p3, p4, p5, p6)

        return ans

    def decode_jpeg_python(self, image_buffer, scope=None):
        with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
            cN = FLAGS.FRAMES_IN_SEG // FLAGS.temporal_downsample_factor
            cH = FLAGS.IM_HEIGHT // FLAGS.decode_downsample_factor
            cW = FLAGS.IM_WIDTH // FLAGS.decode_downsample_factor
            cC = 3
            cDown = FLAGS.decode_downsample_factor
            decoded = tf.py_func(self.decode_batch_libturbo, [image_buffer, cH, cW, cC, cDown], [tf.uint8])[0]
            decoded.set_shape([cN, cH, cW, cC])
            return decoded

    @staticmethod
    def future_smooth(actions, naction, nfuture):
        # TODO: could add weighting differently between near future and far future
        # given a list of actions, for each time step, return the distribution of future actions
        l = len(actions) # action is a list of integers, from 0 to naction-1, negative values are ignored
        out = np.zeros((l, naction), dtype=np.float32)
        for i in range(l):
            # for each output position
            total = 0
            for j in range(min(nfuture, l-i)):
                # for each future position
                # current deal with i+j action
                acti = i + j
                if actions[acti]>=0:
                    out[i, actions[acti]] += 1
                    total += 1
            if total == 0:
                out[i, MyDataset.turn_str2int['straight']] = 1.0
            else:
                out[i, :] = out[i, :] / total
        return out

    @staticmethod
    def speed_to_future_has_stop(speed, nfuture, speed_limit_as_stop):
        # expect the stop_label to be 1 dimensional, representing the stop labels along time
        # nfuture is how many future stop labels to consider
        speed = np.linalg.norm(speed, axis=1)
        stop_label = np.less(speed, speed_limit_as_stop)
        stop_label = stop_label.astype(np.int32)

        # the naction=2 means: the number of valid actions are 2
        smoothed = MyDataset.future_smooth(stop_label, 2, nfuture)
        out = np.less(0, smoothed[:, 1]).astype(np.int32)
        return out

    @staticmethod
    def no_stop_dropout_valid(stop_label, drop_prob):
        nbatch = stop_label.shape[0]
        ntime  = stop_label.shape[1]
        out = np.ones(nbatch, dtype=np.bool)
        for i in range(nbatch):
            # determine whether this seq has stop
            has_stop = False
            for j in range(ntime):
                if stop_label[i, j]:
                    has_stop = True
                    break
            if not has_stop:
                if np.random.rand() < drop_prob:
                    out[i] = False
        
        return out

    @staticmethod
    def speed_to_course(speed):
        pi = math.pi
        if speed[1] == 0:
            if speed[0] > 0:
                course = pi / 2
            elif speed[0] == 0:
                course = None
            elif speed[0] < 0:
                course = 3 * pi / 2
            return course
        course = math.atan(speed[0] / speed[1])
        if course < 0:
            course = course + 2 * pi
        if speed[1] > 0:
            course = course
        else:
            course = pi + course
            if course > 2 * pi:
                course = course - 2 * pi
        assert not math.isnan(course)
        return course

    @staticmethod
    def to_course_list(speed_list):
        l = speed_list.shape[0]
        course_list = []
        for i in range(l):
            speed = speed_list[i,:]
            course_list.append(MyDataset.speed_to_course(speed))
        return course_list

    turn_str2int={'not_sure': -1, 'straight': 0, 'slow_or_stop': 1,
                  'turn_left': 2, 'turn_right': 3,
                  'turn_left_slight': 4, 'turn_right_slight': 5,}
                  #'acceleration': 6, 'deceleration': 7}

    turn_int2str={y: x for x, y in turn_str2int.iteritems()}
    naction = np.sum(np.less_equal(0, np.array(turn_str2int.values())))

    @staticmethod
    def turning_heuristics(speed_list, speed_limit_as_stop=0):
        course_list = MyDataset.to_course_list(speed_list)
        speed_v = np.linalg.norm(speed_list, axis=1)
        l = len(course_list)
        action = np.zeros(l).astype(np.int32)
        course_diff = np.zeros(l).astype(np.float32)

        enum = MyDataset.turn_str2int

        thresh_low = (2*math.pi / 360)*1
        thresh_high = (2*math.pi / 360)*35
        thresh_slight_low = (2*math.pi / 360)*3

        def diff(a, b):
            # return a-b \in -pi to pi
            d = a - b
            if d > math.pi:
                d -= math.pi * 2
            if d < -math.pi:
                d += math.pi * 2
            return d

        for i in range(l):
            if i == 0:
                action[i] = enum['not_sure']
                continue

            # the speed_limit_as_stop should be small,
            # this detect strict real stop
            if speed_v[i] < speed_limit_as_stop + 1e-3:
                # take the smaller speed as stop
                action[i] = enum['slow_or_stop']
                continue

            course = course_list[i]
            prev = course_list[i-1]

            if course is None or prev is None:
                action[i] = enum['slow_or_stop']
                course_diff[i] = 9999
                continue

            course_diff[i] = diff(course, prev)*360/(2*math.pi)
            if thresh_high > diff(course, prev) > thresh_low:
                if diff(course, prev) > thresh_slight_low:
                    action[i] = enum['turn_right']
                else:
                    action[i] = enum['turn_right_slight']

            elif -thresh_high < diff(course, prev) < -thresh_low:
                if diff(course, prev) < -thresh_slight_low:
                    action[i] = enum['turn_left']
                else:
                    action[i] = enum['turn_left_slight']
            elif diff(course, prev) >= thresh_high or diff(course, prev) <= -thresh_high:
                action[i] = enum['not_sure']
            else:
                action[i] = enum['straight']

            if FLAGS.no_slight_turn:
                if action[i] == enum['turn_left_slight']:
                    action[i] = enum['turn_left']
                if action[i] == enum['turn_right_slight']:
                    action[i] = enum['turn_right']

            # this detect significant slow down that is not due to going to turn
            if FLAGS.deceleration_thres > 0 and action[i] == enum['straight']:
                hz = FLAGS.frame_rate / FLAGS.temporal_downsample_factor
                acc_now = (speed_v[i] - speed_v[i - 1]) / (1.0 / hz)
                if acc_now < - FLAGS.deceleration_thres:
                    action[i] = enum['slow_or_stop']
                    continue

        # avoid the initial uncertainty
        action[0] = action[1]
        return action

    @staticmethod
    def turn_future_smooth(speed, nfuture, speed_limit_as_stop):
        # this function takes in the speed and output a smooth future action map
        turn = MyDataset.turning_heuristics(speed, speed_limit_as_stop)
        smoothed = MyDataset.future_smooth(turn, MyDataset.naction, nfuture)
        return smoothed

    @staticmethod
    def fix_none_in_course(course_list):
        l = len(course_list)

        # fix the initial None value
        not_none_value = 0
        for i in range(l):
            if not (course_list[i] is None):
                not_none_value = course_list[i]
                break
        for i in range(l):
            if course_list[i] is None:
                course_list[i] = not_none_value
            else:
                break

        # a course could be None, use the previous course in that case
        for i in range(1, l):
            if course_list[i] is None:
                course_list[i] = course_list[i - 1]
        return course_list

    @staticmethod
    def relative_future_location(speed, nfuture, sample_rate):
        # given the speed vectors, calculate the future location relative to
        # the current location, with facing considered
        course_list = MyDataset.to_course_list(speed)
        course_list = MyDataset.fix_none_in_course(course_list)

        # integrate the speed to get the location
        loc = util_car.integral(speed, 1.0 / sample_rate)

        # project future motion on to the current facing direction
        # this is counter clock wise
        def rotate(vec, theta):
            c = math.cos(theta)
            s = math.sin(theta)
            xp = c * vec[0] - s * vec[1]
            yp = s * vec[0] + c * vec[1]
            return np.array([xp, yp])

        out = np.zeros_like(loc)
        l = out.shape[0]
        for i in range(l):
            future = loc[min(i+nfuture, l-1), :]
            delta = future - loc[i, :]
            out[i, :] = rotate(delta, course_list[i])

        return out

    @staticmethod
    def relative_future_course_speed(speed, nfuture, sample_rate):
        def norm_course_diff(course):
            if course > math.pi:
                course = course - 2*math.pi
            if course < -math.pi:
                course = course + 2*math.pi
            return course

        # given the speed vectors, calculate the future location relative to
        # the current location, with facing considered
        course_list = MyDataset.to_course_list(speed)
        course_list = MyDataset.fix_none_in_course(course_list)

        # integrate the speed to get the location
        loc = util_car.integral(speed, 1.0 / sample_rate)

        out = np.zeros_like(loc)
        l = out.shape[0]
        for i in range(l):
            if i+nfuture < l:
                fi = min(i + nfuture, l - 1)
                # first is course diff
                out[i, 0] = norm_course_diff(course_list[fi] - course_list[i])
                # second is the distance
                out[i, 1] = np.linalg.norm(loc[fi, :] - loc[i, :])
            else:
                # at the end of the video, just use what has before
                out[i,:] = out[i-1,:]

        # normalize the speed to be per second
        timediff = 1.0 * nfuture / sample_rate
        out = out / timediff

        return out


    def decode_png(self, image_buffer, scope=None):
        with tf.op_scope([image_buffer], scope, 'decode_png'):
            # decode PNG
            fn = lambda encoded: tf.image.decode_png(encoded,
                                                      channels=1)

            decoded = tf.map_fn(fn,
                                image_buffer,
                                dtype=tf.uint8,
                                parallel_iterations=10,
                                back_prop=False,
                                swap_memory=False,
                                infer_shape=True,
                                name="map_decode_png")

            # to float, to the range 0.0 - 255.0
            images = tf.cast(decoded, dtype=tf.int32, name="image_to_float")
            tf.image.resize_nearest_neighbor(images, [FLAGS.IM_HEIGHT, FLAGS.IM_WIDTH], align_corners=None, name=None)
            images.set_shape([FLAGS.FRAMES_IN_SEG // FLAGS.temporal_downsample_factor,
                              FLAGS.IM_HEIGHT / FLAGS.decode_downsample_factor,
                              FLAGS.IM_WIDTH / FLAGS.decode_downsample_factor, 1])
            return images
    
    @staticmethod
    def parse_array(array):

        type_code = np.asscalar(np.fromstring(array[0:4], dtype=np.int32))
        shape_size = np.asscalar(np.fromstring(array[4:8], dtype=np.int32))

        shape = np.fromstring(array[8: 8+4 * shape_size], dtype=np.int32)
        if type_code == 5:#cv2.CV_32F:
            dtype = np.float32
        if type_code == 6:#cv2.CV_64F:
            dtype = np.float64
        return np.fromstring(array[8+4 * shape_size:], dtype=dtype).reshape(shape)

    
    def read_array(self, array_buffer):
        fn = lambda array: MyDataset.parse_array(array)
        ctx_decoded = map(fn, array_buffer)                       
        return [ctx_decoded]


    def queue_cityscape(self, image_dir, seg_dir):
        city_im_queue = []
        city_seg_queue = []
        with open(image_dir,'r') as f:
            for content in f.readlines():
                city_im_queue.append(content)
        with open(seg_dir,'r') as f:
            for content in f.readlines():
                city_seg_queue.append(content)

        assert(len(city_im_queue) == len(city_seg_queue))

        return city_im_queue, city_seg_queue

    def read_cityscape(self, city_im_queue, city_seg_queue, read_n):
        global city_pointer

        # get the value of the pointer now
        city_lock.acquire()
        curr_pointer = city_pointer
        city_pointer = (city_pointer + read_n) % len(city_im_queue)
        city_lock.release()

        def read_one_image(path):
            path = path.strip('\n')
            data = misc.imread(path)
            data_resized = misc.imresize(data, [FLAGS.IM_HEIGHT, FLAGS.IM_WIDTH], 'nearest')
            return data_resized

        ans_im = []
        ans_seg = []

        for i in range(read_n):
            pointer_now = (curr_pointer + i) % len(city_im_queue)
            # the color image
            image = read_one_image(city_im_queue[pointer_now])
            image = image.astype("float32")
            ans_im.append(image)

            # the segmentation
            image = read_one_image(city_seg_queue[pointer_now])
            image = image.astype("int32")
            ans_seg.append(image)

        return [ans_im, ans_seg]

    def parse_example_proto(self, example_serialized):#TODO(lowres check this mainly)
        # Dense features in Example proto.
        feature_map = {
                'image/encoded': tf.VarLenFeature(dtype=tf.string),
                'image/speeds': tf.VarLenFeature(dtype=tf.float32),
                'image/class/video_name': tf.FixedLenFeature([1], dtype=tf.string, default_value=''),
        }
        if FLAGS.only_seg == 1:
            feature_map.update({'image/segmentation': tf.VarLenFeature(dtype=tf.string),
                                'image/context': tf.VarLenFeature(dtype=tf.string)})

        if FLAGS.use_speed_yaw:
            feature_map.update({'sensor/yaw_imu': tf.VarLenFeature(dtype=tf.float32),
                                'sensor/speed_steer': tf.VarLenFeature(dtype=tf.float32)})

        features = tf.parse_single_example(example_serialized, feature_map)

        # if the data is downsampled by a temporal factor, the starting point should be random, such that we could use
        # all the data
        if FLAGS.non_random_temporal_downsample:
            tstart = 0
        else:
            tstart = tf.random_uniform([], minval=0, maxval=FLAGS.temporal_downsample_factor, dtype=tf.int32)
        len_downsampled = FLAGS.FRAMES_IN_SEG // FLAGS.temporal_downsample_factor
        if FLAGS.only_seg == 1:
            seg     = features['image/segmentation'].values[:]
            seg.set_shape([len_downsampled])
            ctx     = features['image/context'].values[:]
            ctx.set_shape([len_downsampled])

        name = features['image/class/video_name']

        encoded = features['image/encoded'].values[:FLAGS.FRAMES_IN_SEG]
        encoded_sub = encoded[tstart::FLAGS.temporal_downsample_factor]
        encoded_sub.set_shape([len_downsampled])
        if FLAGS.no_image_input:
            # no image input is used, but the previous steps is done because
            # we assume we have an list of empty image inputs
            decoded = tf.zeros([len_downsampled,
                                FLAGS.IM_HEIGHT / FLAGS.decode_downsample_factor,
                                FLAGS.IM_WIDTH / FLAGS.decode_downsample_factor,
                                3], tf.uint8)
        else:
            decoded = self.decode_jpeg(encoded_sub)
            if FLAGS.only_seg == 1:
                seg_decoded  = self.decode_png(seg)
                ctx_decoded  = tf.py_func(self.read_array,[ctx],[tf.float32])[0]
                ctx_decoded.set_shape([len_downsampled, ctx_channel, ctx_height, ctx_width])

        decoded_raw = decoded
        if FLAGS.resize_images != "":
            # should have format: new_height, new_width
            sp_size = FLAGS.resize_images.split(",")
            assert(len(sp_size) == 2)
            new_size = (int(sp_size[0]), int(sp_size[1]))
            decoded = tf.image.resize_bilinear(decoded, new_size)
            #decoded = tf.image.resize_nearest_neighbor(decoded, new_size)
            decoded = tf.cast(decoded, tf.uint8)

        if FLAGS.crop_car_hood > 0:
            decoded = decoded[:, :-FLAGS.crop_car_hood, :, :]

        speed = features['image/speeds'].values
        speed = tf.reshape(speed, [-1, 2])
        speed = speed[:FLAGS.FRAMES_IN_SEG, :]
        speed = speed[tstart::FLAGS.temporal_downsample_factor, :]
        speed.set_shape([len_downsampled, 2])

        # from speed to stop labels
        stop_label = tf.py_func(self.speed_to_future_has_stop,
                                [speed, FLAGS.stop_future_frames, FLAGS.speed_limit_as_stop],
                                [tf.int32])[0] #TODO(lowres: length of smoothed time)
        stop_label.set_shape([len_downsampled])

        # Note that the turning heuristic is tuned for 3Hz video and urban area
        # Note also that stop_future_frames is reused for the turn
        turn = tf.py_func(self.turn_future_smooth,
                               [speed, FLAGS.stop_future_frames, FLAGS.speed_limit_as_stop],
                               [tf.float32])[0]  #TODO(lowres)
        turn.set_shape([len_downsampled, self.naction])


        if FLAGS.use_speed_yaw:
            yaw = features['sensor/yaw_imu'].values
            spd = features['sensor/speed_steer'].values
            ys = tf.pack([yaw, spd], axis=1, name="stack_yaw_speed")
            # Now the shape is N*2
            ys = ys[tstart : FLAGS.FRAMES_IN_SEG : FLAGS.temporal_downsample_factor, :]
            ys.set_shape([len_downsampled, 2])

            # compute locs from ys
            ys = tf.pad(  ys,
                          [[0, FLAGS.stop_future_frames], [0, 0]],
                          mode="SYMMETRIC",
                          name="pad_afterwards")
            ys = ys[FLAGS.stop_future_frames:, :]
            ys.set_shape([len_downsampled, 2])
            locs = ys
            print("data loader is using raw yaw and speed")
        else:
            # get the relative future location
            # Note that we again abuse the notation a little bit, reusing stop_future_frames
            # TODO: normalize the course and speed by time
            locs = tf.py_func(self.relative_future_course_speed,
                              [speed, FLAGS.stop_future_frames, FLAGS.frame_rate / FLAGS.temporal_downsample_factor],
                              [tf.float32])[0]
            locs.set_shape([len_downsampled, 2])


        # batching one 10 second segments into several smaller segments
        batching_inputs = [decoded, speed, stop_label, turn, locs]
        if FLAGS.only_seg == 1:
            batching_inputs += [seg_decoded, ctx_decoded]
            decoded_raw_loc = 7
        else:
            decoded_raw_loc = 5
        batching_inputs += [decoded_raw]
        batched = [self.batching(x, len_downsampled) for x in batching_inputs]

        name = tf.tile(name, [batched[0].get_shape()[0].value])

        ins = batched[0:2] + [name]
        outs = batched[2:5]
        if FLAGS.city_data:
            # city batch means how many batch does each video sequence forms
            FLAGS.city_batch = len_downsampled // FLAGS.n_sub_frame
            
            # here we want to read in the cityscape data and downsample in the loop
            city_im_queue, city_seg_queue= self.queue_cityscape(FLAGS.city_image_list,
                                                                FLAGS.city_label_list)

            global city_pointer
            city_pointer = 0
            read_n = city_frames * FLAGS.city_batch
            city_im, city_seg = tf.py_func(self.read_cityscape,
                                            [city_im_queue, city_seg_queue, read_n],
                                            [tf.float32, tf.int32])

            city_im = tf.reshape(city_im, [FLAGS.city_batch, city_frames, FLAGS.IM_HEIGHT, FLAGS.IM_WIDTH, city_im_channel])
            city_seg =  tf.reshape(city_seg, [FLAGS.city_batch, city_frames, FLAGS.IM_HEIGHT, FLAGS.IM_WIDTH, city_seg_channel])

            if FLAGS.resize_images != "":
                # should have format: new_height, new_width
                sp_size = FLAGS.resize_images.split(",")
                assert (len(sp_size) == 2)
                new_size = (int(sp_size[0]), int(sp_size[1]))
                city_im = tf.reshape(city_im, [FLAGS.city_batch*city_frames, FLAGS.IM_HEIGHT, FLAGS.IM_WIDTH, city_im_channel])
                city_seg = tf.reshape(city_seg, [FLAGS.city_batch*city_frames, FLAGS.IM_HEIGHT, FLAGS.IM_WIDTH, city_seg_channel])
                city_im = tf.image.resize_bilinear(city_im, new_size)
                city_seg = tf.image.resize_nearest_neighbor(city_seg, new_size)
                city_im = tf.reshape(city_im,
                                     [FLAGS.city_batch, city_frames, new_size[0], new_size[1], city_im_channel])
                city_seg = tf.reshape(city_seg, [FLAGS.city_batch, city_frames, new_size[0], new_size[1],
                                                 city_seg_channel])
            ins += [city_im]
            outs += [city_seg]
        if FLAGS.only_seg == 1:
            ins = ins + batched[5:7]
            outs = outs

        # adding the raw images
        ins += batched[decoded_raw_loc:(decoded_raw_loc+1)]

        if FLAGS.action_mapping_loss:
            assert not FLAGS.city_data
            assert not FLAGS.only_seg
            outs += [ins[-2]]

        # dropout non-stop videos
        if FLAGS.balance_drop_prob > 0:
            retained = tf.py_func(self.no_stop_dropout_valid,
                                  [outs[0], FLAGS.balance_drop_prob],
                                  [tf.bool])[0]
            retained.set_shape([ outs[0].get_shape()[0].value ])

            select = lambda tensors, valid: [util.bool_select(x, valid) for x in tensors]
            ins = select(ins, retained)
            outs = select(outs, retained)
        return ins, outs

    def batching(self, tensor, FRAMES_IN_SEG):
        T = FLAGS.n_sub_frame

        batch_len = FRAMES_IN_SEG // T
        valid_len = batch_len * T
        dim = tensor.get_shape().ndims
        tensor = tf.slice(tensor, [0]*dim, [valid_len] + [-1] * (dim-1))

        # should use tensor.get_shape()[i].value to get a real number
        new_shape = [batch_len, T] + [x.value for x in tensor.get_shape()[1:]]
        tensor = tf.reshape(tensor, new_shape)
        return tensor

    def visualize(self, net_inputs, net_outputs):
        # input a batch of training examples of form [Tensor1, Tensor2, ... Tensor_n]
        # net_inputs: usually images; net_outputs: usually the labels
        # this function visualize the data that is read in, do not return anything but use tf.summary

        # visualize the video using multiple images
        # their is no way to visualize time sequence now, so isvalid and isstop couldn't be visualized
        if not FLAGS.no_image_input:
            decoded = net_inputs[0]
            visualize = tf.cast(decoded[0,:,:,:,:], tf.uint8)
            tf.image_summary("video_seq", visualize, max_images=FLAGS.n_sub_frame)

    def augmentation(self, is_train, net_inputs, net_outputs):
        # augment the network input tensors and output tensors by whether is_train
        # return augment_net_input, augment_net_output
        if FLAGS.use_data_augmentation and is_train:
            # TODO: has not debug this block yet
            images = net_inputs[0]
            with tf.variable_scope("distort_video"):
                print("using random crop and brightness and constrast jittering")
                # shape = B F H W C
                shape = [x.value for x in images.get_shape()]
                images = tf.random_crop(images, [shape[0], shape[1],
                                                 int(shape[2]*0.8),
                                                 int(shape[3]*0.8),
                                                 shape[4]])

                images = tf.image.random_brightness(images, max_delta=32. / 255.)
                images = tf.image.random_contrast(images, lower=0.6, upper=1.4)
                #images = tf.image.random_saturation(images, lower=0.9, upper=1.1)
                #images = tf.image.random_hue(images, max_delta=0.05)

                # The random_* ops do not necessarily clamp. But return uint8 thus not needed
                #images = tf.clip_by_value(images, 0, 255)
                shape_new = tf.shape(images)
                images = tf.reshape(images, [shape[0]*shape[1], shape_new[2], shape_new[3], shape[4]])
                images = tf.image.resize_images(images, [shape[2], shape[3]])
                images = tf.reshape(images, shape)
                images = tf.cast(images, tf.uint8)
            net_inputs[0] = images

        if FLAGS.use_perspective_augmentation:
            images = net_inputs[0] # shape:: N * F * HWC
            images_shape = [x.value for x in images.get_shape()]
            future_labels = net_outputs[2]  # shape: N * F * 2
            future_labels_shape = [x.value for x in future_labels.get_shape()]

            images, future_labels = tf.py_func(MyDataset.perspective_changes,
                                               [images, future_labels],
                                               [tf.uint8, tf.float32])

            images = tf.reshape(images, [images_shape[0]*images_shape[1], images_shape[2], images_shape[3], images_shape[4]])
            images = tf.image.resize_bilinear(images, (228, 228))
            images = tf.cast(images, tf.uint8)
            images = tf.reshape(images, [images_shape[0], images_shape[1], 228, 228, images_shape[4]])

            future_labels.set_shape(future_labels_shape)
            net_inputs[0] = images
            net_outputs[2] = future_labels

        return net_inputs, net_outputs

    @staticmethod
    def generate_meshlist(arange1, arange2):
        return np.dstack(np.meshgrid(arange1, arange2, indexing='ij')).reshape((-1, 2))


    # the input should be bottom cropped image, i.e. no car hood
    @staticmethod
    def rotate_ground(original, theta, horizon=60, half_height=360 / 2, focal=1.0):
        height, width, channel = original.shape
        # the target grids
        yp = range(height - horizon, height)
        xp = range(0, width)

        # from pixel to coordinates
        y0 = (np.array(yp) - half_height) * 1.0 / half_height
        x0 = (np.array(xp) - width / 2) / (width / 2.0)

        # form the mesh
        mesh = MyDataset.generate_meshlist(x0, y0)
        # compute the source coordinates
        st = math.sin(theta)
        ct = math.cos(theta)
        deno = ct * focal + st * mesh[:, 0]
        out = np.array([(-st * focal + ct * mesh[:, 0]) / deno, mesh[:, 1] / deno])

        # interpolate
        vout = []
        for i in range(3):
            f = interpolate.RectBivariateSpline(y0, x0, original[- horizon:, :, i])
            values = f(out[1, :], out[0, :], grid=False)
            vout.append(values)

        lower = np.reshape(vout, (3, width, horizon)).transpose((2, 1, 0)).astype("uint8")

        # compute the upper part
        out = np.reshape(out[0, :], (width, horizon))
        out = out[:, 0]
        f = interpolate.interp1d(x0, original[:-horizon, :, :], axis=1,
                                 fill_value=(original[:-horizon, 0, :], original[:-horizon, -1, :]),
                                 bounds_error=False)
        ans = f(out)
        ans = ans.astype("uint8")

        return np.concatenate((ans, lower), axis=0)


    @staticmethod
    def perspective_changes(images, future_labels):
        N, F, H, W, C = images.shape
        N2, F2, C2 = future_labels.shape
        assert (N == N2)
        assert (F == F2)
        assert (C2 == 2)

        perspective_aug_prob = 0.03
        perspective_recover_time = 2.0 # second
        perspective_theta_std = 0.15
        # image related
        horizon = 60
        half_height = 360 / 2
        focal = 1.0

        # precomputed constants
        downsampled_framerate = FLAGS.frame_rate / FLAGS.temporal_downsample_factor
        num_frames = int(perspective_recover_time * downsampled_framerate)

        for ni in range(N):
            i = 0
            while i+num_frames < F:
                if random.random() < perspective_aug_prob :
                    # then we need to augment the images starting from this one
                    # random sample a rotate angle
                    theta = random.gauss(0, perspective_theta_std)
                    yaw_rate_delta = -theta / perspective_recover_time

                    for j in range(num_frames):
                        # distort each of the frames and yaw rate
                        images[ni, i, :, :, :] = MyDataset.rotate_ground(
                                    images[ni, i, :, :, :], theta*(1-1.0*j/num_frames), horizon, half_height, focal)
                        future_labels[ni, i, 0] += yaw_rate_delta
                        i += 1
                else:
                    i += 1
        return [images, future_labels]
