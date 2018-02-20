from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import util
import math

FLAGS = tf.app.flags.FLAGS
# TODO: current support of detection and segmentation are preliminary
# both of them are not fully support yet

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")

# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --input_queue_memory_factor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")

# with our LRCN model, you want to disable all the data augmentation methods, include both training
# and testing
tf.app.flags.DEFINE_integer('examples_per_shard', 1024,
                            """Empirically how many examples per shard""")
tf.app.flags.DEFINE_boolean('use_MIMO_inputs_pipeline', False,
                            """Use the multiple inputs and multiple outputs reading pipeline""")
tf.app.flags.DEFINE_integer('num_batch_join', 4,
                            """How many batch_join, large enough to avoid single threaded dequeue many""")
tf.app.flags.DEFINE_integer('buffer_queue_capacity_multiply_factor', 1,
                            """the capacity of the buffer queue is defined as factor*FLAGS.num_batch_join""")

tf.app.flags.DEFINE_boolean('shuffle_files_when_train', True,
                            """Whether shuffle tfrecords during the training phase""")



def inputs(dataset, batch_size=None, num_preprocess_threads=None):
  """Generate batches of ImageNet images for evaluation.

  Use this function as the inputs for evaluating a network.

  Note that some (minimal) image preprocessing occurs during evaluation
  including central cropping and resizing of the image to fit the network.

  Args:
    dataset: instance of Dataset class specifying the dataset.
    batch_size: integer, number of examples in batch
    num_preprocess_threads: integer, total number of preprocessing threads but
      None defaults to FLAGS.num_preprocess_threads.

  Returns:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       image_size, 3].
    labels: 1-D integer Tensor of [FLAGS.batch_size].
  """
  if not batch_size:
    batch_size = FLAGS.batch_size

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.device('/cpu:0'):
    tensors = batch_inputs(
        dataset, batch_size, train=False,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=1)

  # tensors is net_inputs, net_outputs
  return tensors

def distorted_inputs(dataset, batch_size=None, num_preprocess_threads=None):
  """Generate batches of distorted versions of ImageNet images.

  Use this function as the inputs for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    dataset: instance of Dataset class specifying the dataset.
    batch_size: integer, number of examples in batch
    num_preprocess_threads: integer, total number of preprocessing threads but
      None defaults to FLAGS.num_preprocess_threads.

  Returns:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       FLAGS.image_size, 3].
    labels: 1-D integer Tensor of [batch_size].
  """
  if not batch_size:
    batch_size = FLAGS.batch_size

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.device('/cpu:0'):
    tensors = batch_inputs(
        dataset, batch_size, train=True,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=FLAGS.num_readers)

  return tensors

def batch_inputs(dataset, batch_size, train, num_preprocess_threads=None,
                 num_readers=1):
  """Contruct batches of training or evaluation examples from the image dataset.

  Args:
    dataset: instance of Dataset class specifying the dataset.
      See dataset.py for details.
    batch_size: integer
    train: boolean
    num_preprocess_threads: integer, total number of preprocessing threads
    num_readers: integer, number of parallel readers

  Returns:
    images: 4-D float Tensor of a batch of images
    labels: 1-D integer Tensor of [batch_size].

  Raises:
    ValueError: if data is not found
  """
  with tf.name_scope('batch_processing'):
    data_files = dataset.data_files()
    if data_files is None:
      raise ValueError('No data files found for this dataset')

    # Create filename_queue
    if train:
      if not FLAGS.shuffle_files_when_train:
          print("not shuffling files during the training phase")

      filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=FLAGS.shuffle_files_when_train,
                                                      capacity=64)
    else:
      filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=False,
                                                      capacity=1)
    if num_preprocess_threads is None:
      num_preprocess_threads = FLAGS.num_preprocess_threads

    # to reduce the num of preprocessing threads, no longer require this
    #if num_preprocess_threads % 4:
    #  raise ValueError('Please make num_preprocess_threads a multiple '
    #                   'of 4 (%d % 4 != 0).', num_preprocess_threads)

    if num_readers is None:
      num_readers = FLAGS.num_readers

    if num_readers < 1:
      raise ValueError('Please make num_readers at least 1')

    # Approximate number of examples per shard.
    examples_per_shard = FLAGS.examples_per_shard
    # Size the random shuffle queue to balance between good global
    # mixing (more examples) and memory use (fewer examples).
    # 1 image uses 299*299*3*4 bytes = 1MB
    # The default input_queue_memory_factor is 16 implying a shuffling queue
    # size: examples_per_shard * 16 * 1MB = 17.6GB
    if train and FLAGS.shuffle_files_when_train:
      min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
      examples_queue = tf.RandomShuffleQueue(
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples,
          dtypes=[tf.string])
    else:
      if FLAGS.shuffle_files_when_train == False:
          print("using non random shuffle queue")

      examples_queue = tf.FIFOQueue(
          capacity=examples_per_shard + 3 * batch_size,
          dtypes=[tf.string])

    # Create multiple readers to populate the queue of examples.
    if num_readers > 1:
      enqueue_ops = []
      for _ in range(num_readers):
        reader = dataset.reader()
        _, value = reader.read(filename_queue)
        enqueue_ops.append(examples_queue.enqueue([value]))

      tf.train.queue_runner.add_queue_runner(
          tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
      example_serialized = examples_queue.dequeue()
    else:
      reader = dataset.reader()
      _, example_serialized = reader.read(filename_queue)


    if FLAGS.use_MIMO_inputs_pipeline:
        # The new convention for images and labels is a generalized one
        # images: all inputs; labels: all outputs that needs to be predicted
        datan=[]
        for thread_id in range(num_preprocess_threads):
            # 1. this function returns multiple input data (could include both labels and images).
            # This will enable more complex models such as LRCN with egomotion inputs to be able
            # to run with this framework
            # 2. The parse_example_proto function could return more than 1 input for one
            # example_serialized.
            # 3. the returned format is a list of tensors [Tensor1, Tensor2,...., Tensor_n],
            # each of the tensor denotes a small batch of one variable.
            # The tensors for the video might be 5-dim, [batch_size, nframes, H, W, C]
            # 4. We expect future data augmentation code to appear in parse_example_proto
            # itself, since inheriently the augmentation is highly dataset dependent.
            # 5. the parse_example_proto return net_input and net_output as two seperate tensor lists
            net_inputs, net_outputs = dataset.parse_example_proto(example_serialized)
            net_inputs, net_outputs = dataset.augmentation(train, net_inputs, net_outputs)
            datan.append(net_inputs + net_outputs)

        # the single thread batch_join dequeue_many operation might be the bottleneck.
        if net_inputs[0].get_shape()[0].value == batch_size:
            print("output batch of parse_example_proto == required batchsize (%d), no batching needed" % batch_size)
            print("Omitting the batch_join queue")
            joins = datan
            one_joined = datan[-1]
        else:
            # this is quite slow, avoid using this
            joins = []
            for i in range(FLAGS.num_batch_join):
                reduced_factor = max(math.ceil(1.0*num_preprocess_threads / FLAGS.num_batch_join), 2)
                one_joined=tf.train.batch_join(tensors_list=datan,
                                                batch_size=batch_size,
                                                capacity= reduced_factor * batch_size,
                                                enqueue_many=True)
                joins.append(one_joined)
            print(FLAGS.num_batch_join, " batch_joins, each of them capacity is, ",
                  reduced_factor*batch_size, " instances", " Warning: using this might be quite slow!")

        # add a buffering queue to remove the dequeue_many time
        capacity = FLAGS.num_batch_join * FLAGS.buffer_queue_capacity_multiply_factor
        print("buffer queue capacity is: ", capacity, " batches")
        buffer_queue = tf.FIFOQueue(
            capacity= capacity,
            dtypes=[x.dtype for x in one_joined],
            shapes = [x.get_shape() for x in one_joined],
            name = "buffer_queue")

        tf.scalar_summary("queue/%s/fraction_of_%d_full" % (buffer_queue.name, capacity),
                       tf.cast(buffer_queue.size(), tf.float32) *
                       (1. / capacity))

        buffer_ops = [buffer_queue.enqueue(join) for join in joins]

        tf.train.queue_runner.add_queue_runner(
            tf.train.queue_runner.QueueRunner(buffer_queue, buffer_ops))
        data_joined = buffer_queue.dequeue()
        # end of buffer queue

        # The CPU to GPU memory transfer still not resolved

        # recover the inputs and outputs
        joined_inputs=data_joined[:len(net_inputs)]
        joined_outputs=data_joined[len(net_inputs):]

        # dataset's visualizer
        # since only the dataset knows how to visualize the data, let the dataset to provide such method
        dataset.visualize(joined_inputs, joined_outputs)
        return joined_inputs, joined_outputs

    else:
        raise ValueError("have to use MIMO input pipeline")
