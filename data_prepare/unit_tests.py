import unittest, image_to_tfrecords, math
import numpy as np

class TestImageToTfrecordsConversion(unittest.TestCase):

    def test_collect_images(self):
        self.assertEqual('foo', 'FOO')

    def test_convert_png_to_jpeg(self):
        self.assertTrue('FOO'.islower())
        self.assertFalse('Foo'.isupper())

    def test_convert_to_tfrecords(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

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
            'The length of the splitted list is {1}, but it should be {0}.'.format(len(random_list)/108, len(splitted_list)))


if __name__ == '__main__':
    unittest.main()
