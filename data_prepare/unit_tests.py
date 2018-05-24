import unittest, image_to_tfrecords, math, os, sys
import numpy as np
import tensorflow as tf
import os, math

class TestImageToTfrecordsConversion(unittest.TestCase):

    def test_collect_images(self):
        list_of_list_names = image_to_tfrecords.collect_images(BASE_FOLDER_PATH)
        image_file_count = 0
        for file in os.listdir(BASE_FOLDER_PATH):
            file_path = os.path.join(BASE_FOLDER_PATH, file)
            if os.path.isdir(file_path):
                image_file_count += len(os.listdir(file_path))
        print("Number of files found: {}\n".format(image_file_count))
        self.assertEqual(int(math.ceil(image_file_count / 108.0)), len(list_of_list_names))

    def test_convert_png_to_jpeg(self):
        pass

    def test_convert_to_tfrecords(self):
        list_of_list_names = image_to_tfrecords.collect_images(BASE_FOLDER_PATH)

        index = 0
        reconstructed_images = list()
        original_images = list()
        for sublist in list_of_list_names:
            original_images = image_to_tfrecords.convert_to_tfrecords(sublist, ".unittest/batch_{}.tfrecords".format(index))
            index += 1

            record_iterator = tf.python_io.tf_record_iterator(path=".unittest/batch_{}.tfrecords".format(index - 1))

            for string_record in record_iterator:

                example = tf.train.Example()
                example.ParseFromString(string_record)

                height = int(example.features.feature['image/height']
                                             .int64_list
                                             .value[0])

                width = int(example.features.feature['image/width']
                                            .int64_list
                                            .value[0])

                img_string = (example.features.feature['image/encoded']
                                              .bytes_list
                                              .value[0])

                img_1d = np.fromstring(img_string, dtype=np.uint8)
                reconstructed_img = img_1d.reshape((height, width, -1))


                reconstructed_images.append(reconstructed_img)


            for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
                self.assertTrue(np.allclose(original_pair, reconstructed_pair))

            reconstructed_images = list()
            original_images = list()


    def test_split_list(self):
        """
            Tests the 'split_list' function in 'image_to_tfrecords.py' to see
            whether it would be able to produce batches consisting of 108 items or less
        """
        random_list = list(np.random.uniform(1, 100, 1000))
        splitted_list = image_to_tfrecords.split_list(random_list, 108)

        for sublist in splitted_list:
            self.assertTrue(len(sublist) <= 108, "It is not less than 108.")

        self.assertEqual(math.ceil(len(random_list)/108.0), len(splitted_list),
            'The length of the splitted list is {1}, but it should be {0}.'.format(
                len(random_list)/108,
                len(splitted_list)))


if __name__ == '__main__':
    """
        To run this unit test, type in the following terminal command:
            python unit_tests.py [base image folder]
    """
    BASE_FOLDER_PATH = None
    if len(sys.argv) == 2:
        BASE_FOLDER_PATH = sys.argv.pop()
    if not os.path.isdir('.unittest'):
        os.mkdir('.unittest')
    unittest.main()
