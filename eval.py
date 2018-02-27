from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime
import math
import os.path
import time
import importlib
import util_car
from util import *
from shutil import copyfile

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import batching
import dataset
import util
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import itertools
import pickle

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")


tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('subsample_factor', 1,
                            """Only evaluate on one out of subsample_factor examples""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' 'test' or 'train'.""")

# Yang: add flags to data provider and model definitions
tf.app.flags.DEFINE_string('data_provider', '',
                           """The data reader class, which is located under ./data_provider/""")
tf.app.flags.DEFINE_string('model_definition', '',
                           """The data reader class""")
dataset_module = importlib.import_module("data_providers.%s" % FLAGS.data_provider)
model = importlib.import_module("models.%s" % FLAGS.model_definition)

tf.app.flags.DEFINE_string('eval_method', 'classification',
                           """The function to evaluate the current task""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """instead of the newest ckpt in dir""")
tf.app.flags.DEFINE_boolean('output_visualizations', False,
                            """Whether to output visualizations beyond testing""")
tf.app.flags.DEFINE_boolean('imagenet_offset', False,
                            """Whether to subtract one from labels to match caffe model""")

tf.app.flags.DEFINE_boolean('use_simplifed_continuous_vis', False,
                            """use a simplified version of visualization""")
tf.app.flags.DEFINE_string('vis_func', "",
                            """When not empty, use this visualization function,
                            this overrides use_simplifed_continuous_vis""")

tf.app.flags.DEFINE_float('sleep_per_iteration', -1.0,
                          '''how long to sleep per iteration, use when fastest evaluation is not necessary''')
tf.app.flags.DEFINE_boolean('save_best_model', False,
                            """save the best model during validation""")

tf.app.flags.DEFINE_string('eval_viz_id', "viz",
                            """the output folder name of the visualization""")

# the best error global recorder
best_error = 1e9
should_save = False
previous_evaluated_model = None
# TODO: detection eval and segmentation eval.

def update_best_error(new_candidate):
  if FLAGS.save_best_model:
    global best_error
    global should_save

    best_error_file = os.path.join(FLAGS.checkpoint_dir, "best_error.txt")
    if os.path.exists(best_error_file):
      with open(best_error_file, "r") as f:
        best_error = float(f.readline().strip())

    if new_candidate < best_error:
      print("found a new better model!! please do not interrupt")
      best_error = new_candidate
      should_save = True

    with open(best_error_file, "w") as f:
      f.write(str(best_error))


def _eval_once(saver, summary_writer, logits_all, labels, loss_op, summary_op, tensors_in):
  """Runs Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    logits: logistic output of the network
    labels: labels
    summary_op: Summary op.
  """
  config = tf.ConfigProto(
    intra_op_parallelism_threads=1)
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if FLAGS.pretrained_model_checkpoint_path:
      ckpt_path = FLAGS.pretrained_model_checkpoint_path
      assert tf.gfile.Exists(ckpt_path)
    elif ckpt and ckpt.model_checkpoint_path:
      ckpt_path = ckpt.model_checkpoint_path
    else:
      print('No checkpoint file found')
      return

    global previous_evaluated_model
    if ckpt_path == previous_evaluated_model:
      print("model %s has been evaluated. Sleep for 2 mins" % ckpt_path)
      time.sleep(120)
      return

    # Restores from checkpoint with absolute path.
    saver.restore(sess, ckpt_path)

    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/imagenet_train/model.ckpt-0,
    # extract global_step from it.
    global_step = ckpt_path.split('/')[-1].split('-')[-1]
    print('Succesfully loaded model from %s at step=%s.' %
          (ckpt_path, global_step))


    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      eval_method = globals()[FLAGS.eval_method]
      summary = eval_method(logits_all, labels, loss_op, sess, coord, summary_op, tensors_in, summary_writer)
      summary_writer.add_summary(summary, global_step)

      previous_evaluated_model = ckpt_path
      # Have finished the evaluation of this round
      global should_save
      if should_save and FLAGS.save_best_model:
        # delete the previous saved best model
        for f in os.listdir(FLAGS.checkpoint_dir):
          if f.endswith(".bestmodel"):
            os.remove(os.path.join(FLAGS.checkpoint_dir, f))
        # save for the current round
        copyfile(ckpt_path, ckpt_path + ".bestmodel")
        should_save = False
        print("saving model finished, you could interrupt now")

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  print(cm)

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, "%.2f" % cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')


def car_discrete(logits_all, labels_in, loss_op, sess, coord, summary_op, tensors_in, summary_writer):
  logits = tf.nn.softmax(logits_all[0])
  labels = labels_in[1] # since the second entry is the turn label
  nclass = labels.get_shape()[-1].value
  labels = tf.reshape(labels, [-1, nclass])

  if FLAGS.city_data:
    city_logits = logits_all[1]
    seg_mask = labels_in[3]
    city_seg_shape = [x.value for x in seg_mask.get_shape()]

    seg_mask = tf.reshape(seg_mask, [-1])
    city_logits = tf.image.resize_nearest_neighbor(city_logits,
                                                   [city_seg_shape[2],
                                                    city_seg_shape[3]])
    city_logits = tf.argmax(city_logits, 3)
    city_logits = tf.reshape(city_logits, [-1])

    weight = tf.less(seg_mask, 255)
    seg_mask = tf.mul(seg_mask, tf.cast(weight,tf.int32))

    mean_iou, iou_update_op = \
      tf.contrib.metrics.streaming_mean_iou(city_logits,
                                            seg_mask,
                                            19,
                                            weights=weight)

  real_loss = tf.nn.softmax_cross_entropy_with_logits(logits_all[0], labels)
  real_loss = tf.reduce_mean(real_loss)

  total_loss = 0.0
  real_acc = 0.0
  logits_all = np.zeros((0, nclass), dtype=np.float32)

  labels_all = np.zeros((0, nclass), dtype=np.float32)

  save_loss = []

  print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
  start_time = time.time()
  summary = tf.Summary()
  need_summary_every_batch = False

  init_op = tf.initialize_local_variables()
  sess.run(init_op)

  num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
  for step in range(num_iter):
    t0 = time.time()
    if coord.should_stop():
      print("coord thinks we should exit, at step: ", step)
      break
    if FLAGS.output_visualizations:
      real_loss_v, loss_v, labels_v, logits_v, tin_out_v = \
          sess.run([real_loss, loss_op, labels, logits, tensors_in+labels_in])
      if FLAGS.use_simplifed_continuous_vis:
        vis_func = util_car.vis_discrete_colormap_antialias
      else:
        vis_func = util_car.vis_discrete

      for isample in range(FLAGS.batch_size):
        vis_func(tin_out_v,
                              logits_v,
                              FLAGS.frame_rate/FLAGS.temporal_downsample_factor,
                              isample,
                              True,
                              os.path.join(FLAGS.eval_dir, FLAGS.eval_viz_id),
                              string_type='image')
      tin_out_v_2 = tin_out_v[2]
    else:
      if FLAGS.city_data:
        real_loss_v, loss_v, labels_v, logits_v, tin_out_v_2, mean_iou_v, iou_update_op_v = \
            sess.run([real_loss, loss_op, labels, logits, tensors_in[2], mean_iou, iou_update_op])
      else:
        real_loss_v, loss_v, labels_v, logits_v, tin_out_v_2 = \
            sess.run([real_loss, loss_op, labels, logits, tensors_in[2]])
    logits_all = np.concatenate((logits_all, logits_v), axis=0)
    labels_all = np.concatenate((labels_all, labels_v), axis=0)
    total_loss = total_loss + loss_v[0]
    real_acc += real_loss_v
    save_loss.append([real_loss_v, tin_out_v_2])

    if step % 20 == 19:
      duration = time.time() - start_time
      sec_per_batch = duration / 20.0
      examples_per_sec = FLAGS.batch_size / sec_per_batch
      print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f sec/batch)' %
            (datetime.now(), step, num_iter, examples_per_sec, sec_per_batch))
      start_time = time.time()

    if need_summary_every_batch:
      summary.ParseFromString(sess.run(summary_op))
      summary_writer.add_summary(summary, 9876)

    tspend = time.time() - t0
    if FLAGS.sleep_per_iteration - tspend > 0:
      time.sleep(FLAGS.sleep_per_iteration - tspend)

  # compute the accuracy, precision, recall, auc, perplexity==loss
  total_loss = total_loss / num_iter
  real_acc /= num_iter
  label1 = np.argmax(labels_all, axis=1)
  pred1  = np.argmax(logits_all, axis=1)
  with open(os.path.join(FLAGS.eval_dir,'seg.pickle'),'w') as f:
      pickle.dump(save_loss, f)

  accuracy = accuracy_score(label1, pred1)
  # each class's L1 diff average
  int2str = dataset_module.MyDataset.turn_int2str
  class_diff = np.sum(np.abs(labels_all - logits_all), axis=0) / labels_all.shape[0]
  class_diff = class_diff.ravel()

  diff_dict = {}
  class_names = []
  for i in range(dataset_module.MyDataset.naction):
    if int2str[i]!="turn_left_slight" and int2str[i]!="turn_right_slight":
      diff_dict[int2str[i]] = class_diff[i]
      class_names.append(int2str[i])

  

  summary.ParseFromString(sess.run(summary_op))
  summary.value.add(tag='test_loss', simple_value=total_loss)
  summary.value.add(tag='test_loss_unbias', simple_value=real_acc)
  update_best_error(real_acc)

  summary.value.add(tag='accuracy', simple_value=accuracy)
  if FLAGS.city_data:
    summary.value.add(tag='meanIOU', simple_value=float(mean_iou_v))

  print("weighted cross entropy=%f, unbias test loss=%f, accuracy=%f, class wise diff:" % (total_loss, real_acc, accuracy))
  for key in diff_dict.keys():
    print("class %s diff = %f" % (key, diff_dict[key]))
    summary.value.add(tag='class_diff/%s' % key, simple_value=np.asscalar(diff_dict[key]))

  # add the confusion matrix
  np.set_printoptions(precision=2)
  cnf_matrix = confusion_matrix(label1, pred1)
  # Plot normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                        title='Normalized Confusion Matrix')
  fig_path = os.path.join(FLAGS.eval_dir, "confusion_matrix.png")
  if os.path.exists(fig_path):
    os.remove(fig_path)
  plt.savefig(fig_path,
              bbox_inches="tight",
              pad_inches=0.3)

  with open(os.path.join(FLAGS.eval_dir, 'seg.pickle'), 'w') as f:
    pickle.dump(save_loss, f)

  return summary

def save_object(object, message):
    from datetime import datetime
    fname = "debug_"+message+"_" + str(datetime.now()).replace(" ", "_") + ".pkl"
    fname = os.path.join(FLAGS.train_dir, fname)
    with open(fname, "wb") as f:
        pickle.dump(object, f)

    return 1

def car_continuous(logits_all_in, labels_in, loss_op, sess, coord, summary_op, tensors_in, summary_writer):
  logits = logits_all_in[0]
  labels = labels_in[2] # since the 3rd entry is the course and speed
  nclass = labels.get_shape()[-1].value # should be 2
  labels = tf.reshape(labels, [-1, nclass])

  total_loss = 0.0

  print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
  start_time = time.time()
  num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
  for step in range(num_iter):
    t0 = time.time()
    if coord.should_stop():
      break
    if FLAGS.output_visualizations:
      loss_v, labels_v, logits_v, tin_out_v = sess.run([loss_op, labels, logits, tensors_in + labels_in])
      # fix the visualization bug
      logits_v_reshape = np.reshape(logits_v, (FLAGS.batch_size, FLAGS.n_sub_frame, -1))
      print("the reshaped logits_v has a shape of:", logits_v_reshape.shape)

      for isample in range(FLAGS.batch_size):

        if FLAGS.vis_func == "":
          if FLAGS.use_simplifed_continuous_vis:
              vis_func = util_car.vis_continuous_interpolated
          else:
              # vis_func = util_car.vis_continuous_simplified    # The too complicated is not readable
              vis_func = util_car.vis_continuous
        else:
          vis_func = getattr(util_car, FLAGS.vis_func)
        vis_func(tin_out_v,
                              logits_v_reshape[isample, :, :],
                              15 / FLAGS.temporal_downsample_factor,
                              model,
                              isample,
                              True,
                              os.path.join(FLAGS.eval_dir, FLAGS.eval_viz_id))

        # save the logits to a file
        names = tin_out_v[2]
        _, short_name = os.path.split(names[isample])
        short_name = short_name.split(".")[0]
        npz_filename = os.path.join(FLAGS.eval_dir, FLAGS.eval_viz_id, short_name + ".logit.npz")
        pickle.dump(logits_v, open(npz_filename, 'wb'))

    else:
      loss_v, labels_v, logits_v = sess.run([loss_op, labels, logits])
    if step == 0:
      logits_all = logits_v
      labels_all = labels_v
    else:
      logits_all = np.concatenate((logits_all, logits_v), axis=0)
      labels_all = np.concatenate((labels_all, labels_v), axis=0)
    total_loss = total_loss + loss_v[0]


    if step % 20 == 19:
      duration = time.time() - start_time
      sec_per_batch = duration / 20.0
      examples_per_sec = FLAGS.batch_size / sec_per_batch
      print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f sec/batch)' %
            (datetime.now(), step, num_iter, examples_per_sec, sec_per_batch))
      start_time = time.time()
    elif step == (num_iter-1):
      print("last round of iter=", step)

    tspend = time.time() - t0
    if FLAGS.sleep_per_iteration - tspend > 0:
      time.sleep(FLAGS.sleep_per_iteration - tspend)

  # compute the accuracy, precision, recall, auc, perplexity==loss
  total_loss = total_loss / num_iter
  #MAPs = model.continous_MAP([logits_all])
  logLikes = model.continous_pdf([logits_all], [labels_all])
  meanLikes = np.mean(logLikes, axis=0)

  summary = tf.Summary()
  summary.ParseFromString(sess.run(summary_op))

  # Warning, now the best metric becomes the average likelihoods, instead of the angular likelihoods
  update_best_error(-np.asscalar(np.mean(meanLikes)))

  if FLAGS.sub_arch_selection == "car_loc_xy":
    summary.value.add(tag='test_loglike/course', simple_value=np.asscalar(meanLikes[0]))
    summary.value.add(tag='test_loglike/speed', simple_value=np.asscalar(meanLikes[1]))
    print("log(likihoods): course=%f, speed=%f" % (meanLikes[0], meanLikes[1]), end='')

  summary.value.add(tag='test_loglike/total', simple_value=np.asscalar(np.mean(meanLikes)))
  summary.value.add(tag='test_loss_biased', simple_value=total_loss)
  print("test cross entropy=%f, log(likelihoods): total=%f" % (total_loss, np.mean(meanLikes)))

  return summary

def evaluate():
  print("in model evaluation")
  dataset = dataset_module.MyDataset(subset=FLAGS.subset)
  assert dataset.data_files()
  FLAGS.num_examples = dataset.num_examples_per_epoch() / FLAGS.subsample_factor

  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels from the dataset.
    tensors_in, tensors_out = batching.inputs(dataset)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = dataset.num_classes() + 1

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits_all = model.inference(tensors_in, num_classes, for_training=False)
    model.loss(logits_all, tensors_out, batch_size=FLAGS.batch_size)
    loss_op = slim.losses.get_losses()

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

    while True:
      _eval_once(saver, summary_writer, logits_all, tensors_out, loss_op, summary_op, tensors_in)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

def main(unused_argv=None):
  if not tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()

if __name__ == '__main__':
  tf.app.run()