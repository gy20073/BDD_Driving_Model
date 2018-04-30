import unittest, image_to_tfrecords, math, os, sys
import numpy as np

class TestImageToTfrecordsConversion(unittest.TestCase):

    def test_collect_images(self):
        self.assertEqual('foo', 'FOO')

    def test_convert_png_to_jpeg(self):
        os.mkdir('.unittest')

        self.assertTrue('FOO'.islower())
        self.assertFalse('Foo'.isupper())

    def test_convert_to_tfrecords(self):
        list_of_list_names = image_to_tfrecords.collect_images(BASE_FOLDER_PATH)

        index = 0
        for sublist in list_of_list_names:
            original_images = image_to_tfrecords.convert_to_tfrecords(sublist, ".unittest/batch_{}.tfrecords".format(index))
            index += 1




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
