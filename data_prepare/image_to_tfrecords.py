from prepare_tfrecords import _int64_feature, _float_feature, _bytes_feature
from PIL import Image
import tensorflow as tf
import numpy as np
import os

def convert_png_to_jpeg(filename):
    """
        Converts an image's format from 'png' to 'jpeg'
    """
    image = Image.open(filename)
    new_filename = filename.replace('.png', '.jpeg')
    image.save(new_filename, "JPEG")
    return new_filename

def convert_to_tfrecords(filename_list, tfrecord_filename='batch.tfrecords'):
    """
        Converts a list of images into TFRecord
    """
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    original_images = list()
    raw_img_list = list()

    for img_path in filename_list:

        img_path = convert_png_to_jpeg(img_path)
        img = np.array(Image.open(img_path))


        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,
        # convert to 1d array and convert to respective
        # shape that image used to have.
        # height = tf.app.flags.image_height
        height = img.shape[0]
        width = img.shape[1]

        # Put in the original images into array
        # Just for future check for correctness
        original_images.append(img)
        img_raw = img.tostring()
        raw_img_list.append(img_raw)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channel': _int64_feature(3),
        'image/class/image_name':_bytes_feature([img_path]),
        'image/format':_bytes_feature(['JPEG']),
        'image/encoded': _bytes_feature(raw_img_list),
        'image/speeds': _float_feature(0),
    }))

    writer.write(example.SerializeToString())

    writer.close()
    return original_images

def split_list(list, size=108):
    """
        Splits the list into batches consisting of 108 PNG images.
        The function returns a list of a sub-lists.
    """
    return [list[i:i+size] for i in range(0, len(list), size)]

def collect_images(base_folder_path):
    """
        Retrieve the list of image names/paths and organize them into batches
        The function returns a list of image batches of a certain size.
    """
    image_path_list = list()

    for folder_name in os.listdir(base_folder_path):
        image_folder_path = os.path.join(base_folder_path, folder_name)
        if not os.path.isdir(image_folder_path):
            continue
        image_batch = list()
        for image_name in os.listdir(image_folder_path):
            if 'final' in image_name and '.png' in image_name:
                image_path = os.path.join(image_folder_path, image_name)
                image_batch.append(image_path)
        image_path_list += split_list(image_batch)

    return image_path_list




if __name__ == '__main__':
    assert tf.app.flags.base_path, 'Need a path to the base folder!'
