from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import shutil
from subprocess import call
import subprocess

import tensorflow as tf
from tensorflow.core.example import example_pb2
from json_to_speed import get_interpolated_speed
from PIL import Image
import numpy as np
import cProfile
import time
import cv2
import scipy.ndimage
import skimage
from scipy.misc import imresize
import matplotlib
import StringIO
import multiprocessing


tf.app.flags.DEFINE_string('video_index', '/data/nx-bdd-20160929/video_filtered_index_38_60_sec.txt', 'filtered video indexing')
tf.app.flags.DEFINE_string('output_directory', '/data/nx-bdd-20160929/tfrecord_fix_speed/', 'Training data directory')

#tf.app.flags.DEFINE_integer('train_shards', 1024, 'Number of shards in training TFRecord files.') 
#tf.app.flags.DEFINE_integer('validation_shards', 128, 'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 16, 'Number of threads to preprocess the images.')
# change truncate_frames when low res
tf.app.flags.DEFINE_integer('truncate_frames', 36*15, 'Number of frames to leave in the saved tfrecords')
tf.app.flags.DEFINE_string('temp_dir_root', '/tmp/', 'the temp dir to hold ffmpeg outputs')

tf.app.flags.DEFINE_boolean('low_res', False, 'the data we want to use is low res')
# constant for the low res resolution
pixelh = 216
pixelw = 384
# constant for the high resolution
HEIGHT = 720
WIDTH = 1280

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def probe_file(filename):
    cmnd = ['ffprobe', '-show_format', '-show_streams', '-pretty', filename]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #print filename
    out, err = p.communicate()
    content = out.split('\n')
    whole_time = 0
    rotate = None
    horizontal = True
    for item in content:
        name = item.split('=')[0]
        tag = name.split(':')
        if tag[0] == 'TAG':
            tag = tag[1]
        if name == 'duration':
            time = item.split(':')
            hour = time[-3].split('=')[1]
            minute = time[-2]
            second = time[-1]
        if name == 'width':
            im_w = int(item.split('=')[1])
        if name == 'height':
            im_h = int(item.split('=')[1])
        if tag == 'rotate':
            rotate = int(item.split('=')[1])
    if im_w <= im_h:
        if rotate is None or rotate == 180 or rotate == -180:
            horizontal = False
    else:
        if rotate == 90 or rotate == -90 or rotate == 270 or rotate == -270:
            horizontal = False
        #print hour, minute, second
    whole_time = float(hour)*3600 + float(minute)*60 + float(second)

    return whole_time, horizontal

def full_im(pixel, all_num):
    # whether this frame is full image or not
    num_l, num_r, num_l_u, num_r_u = all_num

    # num is average pixel intensity over the rest areas
    num = 1.0*(np.sum(pixel) - num_l - num_r - num_l_u - num_r_u) / \
            (pixel.shape[0]*pixel.shape[1]*3 - 4*pixelh*pixelw*3)
    #print(num)
    # return (is a full image)
    return num >= 1

#@profile
def read_one_video(video_path, jobid):
    fd, fprefix, cache_images, out_name = parse_path(video_path, jobid)

    FNULL = open(os.devnull, 'w')
    hz_res = 1 if FLAGS.low_res else 15
    ratio = False

    # save the speed field
    json_path = os.path.join(os.path.dirname(fd), "info", fprefix+".json")
    speeds = get_interpolated_speed(json_path, fprefix+".mov", hz_res)
    if speeds is None:
        # if speed is none, the error message is printed in other functions
        return 0, False

    if speeds.shape[0] < FLAGS.truncate_frames:
        print("skipping since speeds are too short!")
        return 0, False
    
    # filter the too short videos
    duration, ratio = probe_file(video_path)
    if duration < (FLAGS.truncate_frames+1) * 1.0 / hz_res:
        print('the video duration is too short')
        return 0, False

    if abs(speeds.shape[0] - duration*hz_res)>2*hz_res:
        # allow one second of displacement
        print("skipping since unequal speed length and image_list length")
        return 0, False
    
    if not ratio:
        print("the ratio of video is incorrect!", video_path)
        return 0, False

    speeds = speeds[:FLAGS.truncate_frames, :]


    image_list=[]
    if FLAGS.low_res:
        cmnd = ['ffmpeg', 
                '-i', video_path, 
                '-f', 'image2pipe',
                '-loglevel', 'panic', 
                '-pix_fmt','rgb24',
                '-r','1',
                '-vcodec', 'rawvideo', '-']
        pipe = subprocess.Popen(cmnd, stdout = subprocess.PIPE, bufsize=10**7)
        pout, perr = pipe.communicate()
        image_buff = np.fromstring(pout, dtype='uint8')
        if image_buff.size < FLAGS.truncate_frames*HEIGHT*WIDTH*3:
            print(jobid, video_path, image_buff.size, 'Insufficient video size.')
            return 0, False
        image_buff = image_buff[0:FLAGS.truncate_frames*HEIGHT*WIDTH*3]
        image_buff = image_buff.reshape(FLAGS.truncate_frames,HEIGHT,WIDTH,3)

        for i in range(FLAGS.truncate_frames):
            image = image_buff[i,:,:,:]
            image_left = image[HEIGHT-pixelh:HEIGHT, 0:pixelw]
            image_right = image[HEIGHT-pixelh:HEIGHT, WIDTH-pixelw:WIDTH]
            image_left_up = image[0:pixelh, 0:pixelw]
            image_right_up = image[0:pixelh, WIDTH-pixelw:WIDTH]

            all_im = [image_left, image_right, image_left_up, image_right_up]
            all_num = [np.sum(image_left), 
                       np.sum(image_right), 
                       np.sum(image_left_up),
                       np.sum(image_right_up)]
            rank = np.argsort(all_num)
            if i % 5 != 0:
                img = all_im[rank[-1]]
            else:
                full = full_im(image, all_num)

                if full:
                    # saving of full images are commented out
                    #output = StringIO.StringIO()
                    #orig_img = Image.fromarray(image)
                    #orig_img.save(output, format='JPEG',quality=50)
                    #orig_contents = output.getvalue()
                    #output.close()
                    #image_list_orig.append(orig_contents)

                    img = imresize(image, (pixelh, pixelw, 3), interp='nearest')
                else:
                    print(video_path, 'It is not full screen at frames % 5 == 0')
                    return 0, False

            # RGB to BGR
            img=img[:,:,[2,1,0]]
            st=cv2.imencode(".JPEG", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
            # convert the tuple to the second arg, which is the image content
            contents=st[1].tostring("C")

            image_list.append(contents)
    else:
        # generate the video to images to this dir
        if os.path.exists(cache_images):
            shutil.rmtree(cache_images)
        os.mkdir(cache_images)

        call(['ffmpeg',
            '-i', video_path,
            '-r', '15',
            '-qscale:v', '10',
            '-s', '640*360',
            '-threads', '4',
            cache_images + '/%04d.jpg'],
            stdout=FNULL,
            stderr=FNULL)

        for subdir, dirs, files in os.walk(cache_images):
            for f in sorted(files):
                with open(os.path.join(subdir, f), 'r') as f:
                    image_data = f.read()
                    image_list.append(image_data)

        if len(image_list)<FLAGS.truncate_frames:
            print('Insufficient video size.')
            return 0, False
        image_list = image_list[0:FLAGS.truncate_frames]


    if FLAGS.low_res:
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': _int64_feature(pixelh),
            'image/width': _int64_feature(pixelw),
            'image/channel': _int64_feature(3),
            'image/full_height': _int64_feature(HEIGHT),
            'image/full_width': _int64_feature(WIDTH),
            'image/class/video_name':_bytes_feature([video_path]),
            'image/format':_bytes_feature(['JPEG']),
            'image/encoded': _bytes_feature(image_list),
            #'image/encoded_orig': _bytes_feature(image_list_orig),
            'image/speeds': _float_feature(speeds.ravel().tolist()), # ravel l*2 into list
        }))
    else:
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': _int64_feature(360),
            'image/width': _int64_feature(640),
            'image/channel': _int64_feature(3),
            'image/class/video_name':_bytes_feature([video_path]),
            'image/format':_bytes_feature(['JPEG']),
            'image/encoded': _bytes_feature(image_list),
            'image/speeds': _float_feature(speeds.ravel().tolist()), # ravel l*2 into list
        }))

    print(video_path)
    return example, True

def parse_path(video_path, jobid):
    fd, fname = os.path.split(video_path)
    fprefix = fname.split(".")[0]
    cache_images = os.path.join(FLAGS.temp_dir_root, "prepare_tfrecords_image_temp_"+str(jobid))
    out_name = os.path.join(FLAGS.output_directory, fprefix+".tfrecords")
    
    # return all sorts of info: 
    # video_base_path, video_name_wo_prefix, cache_path, out_tfrecord_path
    return (fd, fprefix, cache_images, out_name)

#@profile
def convert_one(video_path, jobid):
    fd, fprefix, cache_images, out_name = parse_path(video_path, jobid)
    if not os.path.exists(out_name):
        example, state = read_one_video(video_path, jobid)
        if state:
            writer = tf.python_io.TFRecordWriter(out_name)
            writer.write(example.SerializeToString())
            writer.close()   

def p_convert(video_path_list, jobid):
    #start = time.time()
    for video_path in video_path_list:
        fd, fprefix, cache_images, out_name = parse_path(video_path, jobid)
        mod = int(fprefix[0:3], 16) % FLAGS.num_threads
        if mod == jobid:
            convert_one(video_path, jobid)
            #end = time.time()
            #if end-start > 60:
            #    break

def parallel_run():
    with open(FLAGS.video_index) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    threads = []
    for i in range(FLAGS.num_threads):
        # arguments are (the range to process, train phase or test phase)
        args = (content, i)
        t = multiprocessing.Process(target=p_convert, args=args)
        #t = threading.Thread(target=p_convert, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('Finished processing all files')
    sys.stdout.flush()

if __name__ == '__main__':
    if FLAGS.low_res:
        print("Warning: using low res specific settings")
    if not tf.gfile.Exists(FLAGS.output_directory):
        tf.gfile.MakeDirs(FLAGS.output_directory)
    parallel_run()

    '''
    sample_video = "/data/nx-bdd-20160929/ride/ba1145b664fc4b84a87ee404427057bd/c18eb46f-7916-41dc-bee8-12b7619821cc.mov"
    handle=lambda : convert_one(sample_video)

    t = timeit.Timer("handle()", "from __main__ import handle")
    print(t.timeit(1))
    '''
