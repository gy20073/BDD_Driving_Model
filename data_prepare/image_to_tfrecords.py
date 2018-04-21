from .prepare_tfrecords import _int64_feature, _float_feature, _bytes_feature
from PIL import Image
import tensorflow as tf
import numpy as np

SAMPLE_FILENAME_PAIRS = [("GTA_dataset/1518144524486_final.png", "GTA_dataset/1518144524486_id.png"),
        ("GTA_dataset/1518144525001_final.png", "GTA_dataset/1518144525001_id.png"),
        ("GTA_dataset/1518144525504_final.png", "GTA_dataset/1518144525504_id.png")]

def convert_to_tfrecords(filename_pairs, tfrecord_filename='convertedImage.tfrecords'):
    """
        Converts an image pairs to TFRecord
    """
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    original_images = []

    for img_path, annotation_path in filename_pairs:

        img = np.array(Image.open(img_path))
        annotation = np.array(Image.open(annotation_path))

        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,
        # convert to 1d array and convert to respective
        # shape that image used to have.
        height = img.shape[0]
        width = img.shape[1]

        # Put in the original images into array
        # Just for future check for correctness
        original_images.append((img, annotation))

        img_raw = img.tostring()
        annotation_raw = annotation.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw)}))

        writer.write(example.SerializeToString())

    writer.close()
