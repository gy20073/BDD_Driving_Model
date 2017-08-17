from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import importlib
import util_car

import numpy as np
import tensorflow as tf

import batching
import dataset
import util
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('subsample_factor', 1,
                            """Only evaluate on one out of subsample_factor examples""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")

# Yang: add flags to data provider and model definitions
tf.app.flags.DEFINE_string('data_provider', '',
                           """The data reader class, which is located under ./data_provider/ """)
tf.app.flags.DEFINE_string('model_definition', '',
                           """The model class""")
dataset_module = importlib.import_module("data_providers.%s" % FLAGS.data_provider)
model = importlib.import_module("models.%s" % FLAGS.model_definition)

tf.app.flags.DEFINE_string('eval_method', 'stat_labels',
                           """The function to evaluate the current task""")
tf.app.flags.DEFINE_string('stat_output_path', '',
                           """Directory where to write stat out""")
tf.app.flags.DEFINE_boolean('stat_datadriven_only', False, 'whether only care about the data driven stats')

def stat_labels(labels_in, sess, coord, tensors_in):
    labels_stop = labels_in[0]  # shape: N * F
    # reshape to 1 dimension
    labels_stop = tf.reshape(labels_stop, [-1])

    discrete_labels = labels_in[1]  # shape: N * F * nclass
    # reshape to 2 dimension
    num_classes = discrete_labels.get_shape()[-1].value
    discrete_labels = tf.reshape(discrete_labels, [-1, num_classes])

    # it's a N * F * 2 tensor
    future_labels = labels_in[2]
    num_classes = future_labels.get_shape()[-1].value
    future_labels = tf.reshape(future_labels, [-1, num_classes])

    if not FLAGS.stat_datadriven_only:
        # up to now, each of them is NF * Nbins tensor
        dense_course, dense_speed = tf.py_func(model.call_label_to_dense_smooth,
                                               [future_labels],
                                               [tf.float32, tf.float32])

    # TODO get the joint stat

    stop_acc = np.array([0, 0])

    discrete_acc = None

    course_acc = None
    speed_acc = None
    count = 0
    future_acc = None

    print('%s: starting getting statistics on (%s).' % (datetime.now(), FLAGS.subset))
    start_time = time.time()
    num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
    for step in range(num_iter):
        if coord.should_stop():
            break

        if not FLAGS.stat_datadriven_only:
            discrete_v, dc, ds, labels_stop_v, future_labels_v = \
                sess.run([discrete_labels, dense_course, dense_speed, labels_stop,future_labels])
            dc = np.mean(dc, axis=0)
            ds = np.mean(ds, axis=0)
        else:
            discrete_v, labels_stop_v, future_labels_v = \
                sess.run([discrete_labels, labels_stop, future_labels])
        discrete_v = np.mean(discrete_v, axis=0)

        if step == 0:
            discrete_acc = discrete_v
            if not FLAGS.stat_datadriven_only:
                course_acc = dc
                speed_acc = ds
            future_acc = future_labels_v
        else:
            discrete_acc += discrete_v
            if not FLAGS.stat_datadriven_only:
                course_acc += dc
                speed_acc += ds
            future_acc = np.concatenate((future_acc, future_labels_v), axis=0)

        for l in labels_stop_v:
            stop_acc[l] += 1

        count += 1

        if step % 20 == 19:
            duration = time.time() - start_time
            sec_per_batch = duration / 20.0
            examples_per_sec = FLAGS.batch_size / sec_per_batch
            print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f sec/batch)' %
                  (datetime.now(), step, num_iter, examples_per_sec, sec_per_batch))
            start_time = time.time()

    discrete_acc /= count
    if not FLAGS.stat_datadriven_only:
        course_acc /= count
        speed_acc /= count

    stop_acc = 1.0 * stop_acc / np.sum(stop_acc)

    np.save(FLAGS.stat_output_path + "_stop", stop_acc)
    np.save(FLAGS.stat_output_path + "_discrete", discrete_acc)
    if not FLAGS.stat_datadriven_only:
        np.save(FLAGS.stat_output_path + "_continuous", (course_acc, speed_acc))
    np.save(FLAGS.stat_output_path + "_dataDriven", future_acc)

    return None


def evaluate():
    dataset = dataset_module.MyDataset(subset=FLAGS.subset)
    assert dataset.data_files()
    FLAGS.num_examples = dataset.num_examples_per_epoch() / FLAGS.subsample_factor

    output_dir = os.path.dirname(FLAGS.stat_output_path)
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        # Get images and labels from the dataset.
        tensors_in, tensors_out = batching.inputs(dataset)

        config = tf.ConfigProto(
            intra_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess,
                                                     coord=coord,
                                                     daemon=True,
                                                     start=True))

                eval_method = globals()[FLAGS.eval_method]
                eval_method(tensors_out, sess, coord, tensors_in)

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

def main(unused_argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()
