from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os, pickle
import re
import time
import importlib
import sys

import numpy as np
import tensorflow as tf

import batching
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tensorflow.python.client import timeline
import util
# populate the --data_dir flag
import dataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                           """How many gpus to use on the system"""
                           """Should be used with CUDA_VISIBLE_DEVICES""")

tf.app.flags.DEFINE_boolean('background_class', True,
                            """Whether to reserve 0 as background.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('training_step_offset', 0,
                            """Subtract offset from global step when calculate 
                            learning rate.
                            It is useful for fine tuning a network.""")

# Yang: delete the fine tuning option here
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

# Yang: add flags to data provider and model definitions
tf.app.flags.DEFINE_string('data_provider', '',
                           """The data reader class, which is located """
                           """under the folder ./data_providers/ """)
tf.app.flags.DEFINE_string('model_definition', '',
                           """The data reader class, located at ./models/""")

model = importlib.import_module("models.%s" % FLAGS.model_definition)
dataset_module = importlib.import_module("data_providers.%s" % FLAGS.data_provider)

tf.app.flags.DEFINE_boolean('profile', False,
                            """Whether to profile using time line object.""")
tf.app.flags.DEFINE_float('clip_gradient_threshold', -1.0,
                            """If the gradient is larger than this value, then clip it.
                            Only valid when > 0""")

tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")

# add a flag to switch optimizer
tf.app.flags.DEFINE_string('optimizer', 'sgd',
                           '''Select which optimizer to use. Currently'''
                           '''available optimizers are sgd and rmsprop''')

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

tf.app.flags.DEFINE_float('momentum', 0.9,
                          """Momentum for SGD or RMSProp.""")

# add display interval flags
tf.app.flags.DEFINE_integer('display_loss', 10,
                            '''display loss info per this batch''')
tf.app.flags.DEFINE_integer('display_summary', 100,
                            '''display tensorboard summary per this batch''')
tf.app.flags.DEFINE_integer('checkpoint_interval', 5000,
                            '''checkpoint per this batch''')

tf.app.flags.DEFINE_string('EWC', 'off',
                           '''Elastic Weight Consolidation method status: off, stat, apply''')

def _tower_loss(inputs, outputs, num_classes, scope):
  # inputs and outputs are two lists of tensors

  """Calculate the total loss on a single tower running the CNN model.

  We perform 'batch splitting'. This means that we cut up a batch across
  multiple GPU's. For instance, if the batch size = 32 and num_gpus = 2,
  then each tower will operate on an batch of 16 images.

  Args:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       FLAGS.image_size, 3].
    labels: Tensor of labels. Shape could be different for each task.
      Classification: 1-D integer Tensor of [batch_size].
      Detection: list of batch_size Tensors where each of them being
        a tuple of ([num_boxes, 4], [num_boxes]) denoting the box coordinates
        and the labels for each box
      Segmentation: a Tensor of [batch_size, image_sz, image_sz]
    num_classes: number of classes
    scope: unique prefix string identifying the ImageNet tower, e.g.
      'tower_0'.

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  logits = model.inference(inputs, num_classes, for_training=True, scope=scope)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  split_batch_size = inputs[0].get_shape().as_list()[0]
  model.loss(logits, outputs, batch_size=split_batch_size)

  # Assemble all of the losses for the current tower only.
  #losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)
  losses = slim.losses.get_losses(scope)

  # Calculate the total loss for the current tower.
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

  total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on TensorBoard.
    loss_name = re.sub('%s_[0-9]*/' % model.TOWER_NAME, '', l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(loss_name +' (raw)', l)
    tf.scalar_summary(loss_name +' (ave)', loss_averages.average(l))

  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss


def _average_gradients(tower_grads, include_square=False):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  average_grads_square = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []

    none_count = 0
    for g, v in grad_and_vars:
      if g == None:
          none_count = none_count + 1
          continue
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    if none_count==0:
        # Average over the 'tower' dimension.
        grad_cat = tf.concat(0, grads)
        grad = tf.reduce_mean(grad_cat, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

        if include_square:
            grad2 = tf.mul(grad_cat, grad_cat, name="square_gradient")
            grad2 = tf.reduce_mean(grad2, 0)
            average_grads_square.append((grad2, v))

    elif none_count == len(grad_and_vars):
        print("None gradient for %s" % (grad_and_vars[0][1].op.name))
    else:
        raise ValueError("None gradient error")
  if include_square:
      return average_grads, average_grads_square
  else:
      return average_grads


def _tensor_list_splits(tensor_list, nsplit):
    # output is nsplit lists, where each one is a list of those tensors
    out = [[] for _ in range(nsplit)]
    # this has T tensors
    for tensor in tensor_list:
        # this has N splits
        sp = tf.split(0, nsplit, tensor)
        for i, split_item in enumerate(sp):
            out[i].append(split_item)
    return out

def train():
  dataset = dataset_module.MyDataset(subset=FLAGS.subset)
  #assert dataset.data_files()

  """Train on dataset for a number of steps."""
  # use gpu:0 instead of cpu0, to avoid RNN GPU variable uninitialized problem
  with tf.Graph().as_default(), tf.device('/gpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    
        
    # Calculate the learning rate schedule.
    num_batches_per_epoch = (dataset.num_examples_per_epoch() /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)


    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step-FLAGS.training_step_offset,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    if FLAGS.optimizer == "rmsprop":
      opt = tf.train.RMSPropOptimizer(lr, decay=RMSPROP_DECAY,
                                      momentum=FLAGS.momentum,
                                      epsilon=RMSPROP_EPSILON)
    elif FLAGS.optimizer == "sgd":
      opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum,
                                       use_nesterov=False)
    elif FLAGS.optimizer == "adadelta":
      opt = tf.train.AdadeltaOptimizer()
    elif FLAGS.optimizer == "adam":
      opt = tf.train.AdamOptimizer()
    else:
      print("optimizer invalid: %s" % FLAGS.optimizer)
      return

    # Get images and labels for ImageNet and split the batch across GPUs.
    assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
        'Batch size must be divisible by number of GPUs')
    split_batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)

    # Override the number of preprocessing threads to account for the increased
    # number of GPU towers.
    #num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
    # choose not to overide, to have a finer control of how many threads to use
    num_preprocess_threads = FLAGS.num_preprocess_threads

    net_inputs, net_outputs = batching.distorted_inputs(
        dataset,
        num_preprocess_threads=num_preprocess_threads)

    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

    init_op = tf.initialize_all_variables()

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    if FLAGS.background_class:
        num_classes = dataset.num_classes() + 1
    else:
        num_classes = dataset.num_classes()
    
     # Split the batch of images and labels for towers.
    # TODO: this might become invalid if we are doing detection
    input_splits = _tensor_list_splits(net_inputs, FLAGS.num_gpus)
    output_splits = _tensor_list_splits(net_outputs, FLAGS.num_gpus)

    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%s' % i):
        with tf.name_scope('%s_%d' % (model.TOWER_NAME, i)) as scope:
          if True:
          # I don't see any improvements by pinning all variables on CPU, so I disabled this
          # Force all Variables to reside on the CPU.
          #with slim.arg_scope([slim.variable], device='/cpu:0'):

          # do not use this line, as it will assign all operations to cpu
          #with tf.device('/cpu:0'):

            # Calculate the loss for one tower of the CNN model. This
            # function constructs the entire CNN model but shares the
            # variables across all towers.
            loss = _tower_loss(input_splits[i], output_splits[i], num_classes,
                               scope)

            if i==0:
                # set different learning rates for different variables
                if hasattr(model, 'learning_rate_multipliers'):
                    # this function returns a dictionary of [varname]=multiplier
                    # learning rate multiplier that equals to one is set by default
                    multiplier = model.learning_rate_multipliers()

                    # computing the vars that needs gradient
                    grad_var_list = []
                    for t in tf.trainable_variables():
                        v = t.op.name
                        if (v in multiplier) and (abs(multiplier[v]) < 1e-6):
                            pass
                        else:
                            grad_var_list.append(t)
                    print("-"*40 + "\n gradient will be computed for vars:")
                    for x in grad_var_list:
                        print(x.op.name)
                else:
                    multiplier = None
                    grad_var_list = None

          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()

          # Retain the summaries from the final tower.
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          # Retain the Batch Normalization updates operations only from the
          # final tower. Ideally, we should grab the updates from all towers
          # but these stats accumulate extremely fast so we can ignore the
          # other stats from the other towers without significant detriment.
          batchnorm_updates = tf.get_collection(ops.GraphKeys.UPDATE_OPS, scope)
          #batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
          #                                      scope)

          # Calculate the gradients for the batch of data on this CNN
          # tower.
          grads = opt.compute_gradients(loss, var_list=grad_var_list)


          # Keep track of the gradients across all towers.
          tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    if FLAGS.EWC == "stat":
        grads, grads2 = _average_gradients(tower_grads, True)
        # merge grads2 into a dict of variable
        out = {}
        vard = {}
        for g2, v in grads2:
            out[v.op.name] = g2
            vard[v.op.name] = v
        grads2 = out
    else:
        grads = _average_gradients(tower_grads)

    # Add a summaries for the input processing and global_step.
    summaries.extend(input_summaries)

    # Add a summary to track the learning rate.
    summaries.append(tf.scalar_summary('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(
            tf.histogram_summary(var.op.name + '/gradients', grad))

    if multiplier:
        print("-" * 40 + "\nusing learning rate multipliers")
        grads_out=[]
        for g, v in grads:
            v_name = v.op.name
            if v_name in multiplier:
                g_out=tf.mul(multiplier[v_name], g)
                print(v_name, " * ", multiplier[v_name])
            else:
                g_out=g
                print(v_name, " * 1.00")
            grads_out.append((g_out, v))
        grads = grads_out

    # gradient clipping
    if FLAGS.clip_gradient_threshold > 0:
        print("-"*40 + "\n Gradient Clipping On")
        t_list = [x[0] for x in grads]
        t_list, gnorm = tf.clip_by_global_norm(
                               t_list,
                               FLAGS.clip_gradient_threshold,
                               name='gradient_clipping')
        grads = [(t_list[i], grads[i][1]) for i in range(len(t_list))]

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.histogram_summary(var.op.name, var))

    # Track the moving averages of all trainable variables.
    # Note that we maintain a "double-average" of the BatchNormalization
    # global statistics. This is more complicated then need be but we employ
    # this for backward-compatibility with our previous models.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY, global_step)

    # Another possiblility is to use tf.slim.get_variables().
    variables_to_average = (tf.trainable_variables() +
                            tf.moving_average_variables())
    variables_averages_op = variable_averages.apply(variables_to_average)

    # Group all updates to into a single train op.
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, variables_averages_op,
                        batchnorm_updates_op)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.merge_summary(summaries)

    # some variables allocated for the accumulators
    if FLAGS.EWC == "stat":
        grads2_accu = {}
        accu_ops = {}
        with tf.device('/gpu:0'):
            for key in grads2.keys():
                shape = [x.value for x in grads2[key].get_shape()]
                grads2_accu[key] = tf.Variable(initial_value=np.zeros(shape, dtype=np.float32),
                                               trainable=False,
                                               name=key+"_accumulator")
                accu_ops[key] = tf.assign_add(grads2_accu[key], grads2[key], name=key+"_assign_add")

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement,
                intra_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.run(init)



    # TODO: not supported to load from different number of towers now
    if FLAGS.pretrained_model_checkpoint_path:
      assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
      #variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
      variables_to_restore = slim.get_variables_to_restore()

      # only restore those that are in the checkpoint
      existing_vars = util.tensors_in_checkpoint_file(FLAGS.pretrained_model_checkpoint_path)
      restore_new = []
      ignore_vars = []
      for x in variables_to_restore:
        if x.op.name in existing_vars:
            restore_new.append(x)
        else:
            ignore_vars.append(x.op.name)
      if len(ignore_vars)>0:
          print("-"*40+"\nWarning: Some variables does not exists in the checkpoint, ignore them: ")
          for x in ignore_vars:
              print(x)
      variables_to_restore = restore_new

      restorer = tf.train.Saver(variables_to_restore)
      restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
      print('%s: Pre-trained model restored from %s' %
            (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
    

    summary_writer = tf.train.SummaryWriter(
        FLAGS.train_dir,
        graph_def=sess.graph.as_graph_def(add_shapes=True))

    start_time = time.time()
    duration_compute=0
    grads2_count = 0

    step_start = int(sess.run(global_step))
    try:
        for step in xrange(step_start, FLAGS.max_steps):
          # call a function in the model definition to do some extra work
          if hasattr(model, 'update_each_step'):
              model.update_each_step(sess, step)

          if FLAGS.EWC == "stat":
              grads2_accu_op = grads2_accu
              if step == (FLAGS.max_steps - 1):
                  sessout = sess.run([grads2_accu_op, accu_ops])
                  grads2_accu=sessout[0]
              else:
                  if FLAGS.profile:
                      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                      run_metadata = tf.RunMetadata()
                      sess.run(accu_ops, options=run_options, run_metadata=run_metadata)

                      tl = timeline.Timeline(run_metadata.step_stats)
                      ctf = tl.generate_chrome_trace_format()
                      with open(os.path.join(FLAGS.train_dir, 'timeline.json'), 'w') as f:
                          f.write(ctf)
                      print("generated a time line profile for one session")
                  else:
                    sess.run(accu_ops)

              grads2_count += 1

              if step == (FLAGS.max_steps - 1):
                  # save the fisher infomation matirx
                  for key in grads2_accu.keys():
                      grads2_accu[key] /= grads2_count
                  fname = os.path.join(FLAGS.train_dir, "EWC_stat.pkl")
                  pickle.dump(grads2_accu, open(fname, "wb"))

                  # save the MAP file
                  vard_v = sess.run(vard)
                  fname = os.path.join(FLAGS.train_dir, "EWC_map.pkl")
                  pickle.dump(vard_v, open(fname, "wb"))

              if (step + 1) % FLAGS.display_loss == 0:
                  print ("processed ", step-step_start, " examples")
              continue

          has_run_meta = False
          if FLAGS.profile:
              run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
              run_metadata = tf.RunMetadata()
              start_time_compute = time.time()
              _, loss_value = sess.run([train_op, loss], options=run_options, run_metadata=run_metadata)
              duration_compute = duration_compute + time.time() - start_time_compute

              # Create the Timeline object, and write it to a json
              tl = timeline.Timeline(run_metadata.step_stats)
              ctf = tl.generate_chrome_trace_format()
              with open(os.path.join(FLAGS.train_dir, 'timeline.json'), 'w') as f:
                  f.write(ctf)
              print("generated a time line profile for one session")
          else:
              start_time_compute = time.time()
              if (step + 1) % (FLAGS.display_summary * 10) == 0:
                  has_run_meta = True
                  # profile in a longer interval
                  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                  run_metadata = tf.RunMetadata()
                  _, loss_value, summary_str = \
                                        sess.run([train_op, loss, summary_op],
                                        options=run_options,
                                        run_metadata=run_metadata)
                  summary_writer.add_run_metadata(run_metadata, 'step%d' % step)
                  summary_writer.add_summary(summary_str, step)
                  print('Adding run metadata for', step)

                  # Create the Timeline object, and write it to a json
                  tl = timeline.Timeline(run_metadata.step_stats)
                  ctf = tl.generate_chrome_trace_format()
                  with open(os.path.join(FLAGS.train_dir, 'timeline.json'), 'w') as f:
                      f.write(ctf)
                  print("generated a time line profile for one session")
              else:
                  _, loss_value = sess.run([train_op, loss])
              duration_compute = duration_compute + time.time() - start_time_compute

          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

          if (step+1) % FLAGS.display_loss == 0:
            duration = (time.time() - start_time) / FLAGS.display_loss
            duration_compute = duration_compute / FLAGS.display_loss

            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch; compute %.1f examples/sec)')
            print(format_str % (datetime.now(), step, loss_value,
                                examples_per_sec, duration,
                                FLAGS.batch_size/duration_compute))
            duration_compute=0
            start_time = time.time()

          if (step+1) % FLAGS.display_summary == 0 and not has_run_meta:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

          # Save the model checkpoint periodically.
          if step % FLAGS.checkpoint_interval == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)

    except KeyboardInterrupt:
        print("Control C pressed. Saving model before exit. ")
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=global_step)
        sys.exit()

def main(_):
  print(tf)
  # fix the profile lib not found issue
  if "LD_LIBRARY_PATH" not in os.environ:
      os.environ["LD_LIBRARY_PATH"] = ""
  os.environ["LD_LIBRARY_PATH"] += os.pathsep + "/usr/local/cuda/extras/CUPTI/lib64"

  if FLAGS.pretrained_model_checkpoint_path != "":
    print("resume training from saved model: % s" % FLAGS.pretrained_model_checkpoint_path)
  elif tf.gfile.Exists(FLAGS.train_dir):
    # find the largest step number: -??
    max_step = -1
    for f in os.listdir(FLAGS.train_dir):
        m = re.search('^model.ckpt-([\d]+)$', f)
        if m:
            found = int(m.group(1))
            if found > max_step:
                max_step = found
    if max_step >= 0:
        ckpt = os.path.join(FLAGS.train_dir, 'model.ckpt-%d' % max_step)
        FLAGS.pretrained_model_checkpoint_path = ckpt
        print("resume training from saved model: % s" % ckpt)
  else:
    tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
