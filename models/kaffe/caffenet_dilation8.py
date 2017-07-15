from models.kaffe.network import Network
import tensorflow as tf

class CaffeNet_dilation8(Network):
     @staticmethod
     def preprocess(images, padding=86):
          images = Network.preprocess(images)

          # TODO: use a more reasonable padding amount
          images = tf.pad(images,
                          [[0, 0], [padding, padding], [padding, padding], [0, 0]],
                          mode="REFLECT",
                          name="padding")
          return images

     def setup(self):
        (self.feed('input')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .lrn(2, 2e-05, 0.75, name='norm1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .lrn(2, 2e-05, 0.75, name='norm2')
             .conv(3, 3, 384, 1, 1, name='conv3', rate=2)
             .conv(3, 3, 384, 1, 1, group=2, name='conv4', rate=2)
             .conv(3, 3, 256, 1, 1, group=2, name='conv5', rate=2)
             .conv(6, 6, 4096, 1, 1, padding='VALID', name='fc6', rate=4)
             .dropout(0.5, name="drop6")
             .conv(1, 1, 4096, 1, 1, padding='VALID', name='fc7', rate=1)
             .dropout(0.5, name="drop7")
             .conv(1, 1, 1000, 1, 1, padding='VALID', name='fc8', rate=1, relu=False)
         )
