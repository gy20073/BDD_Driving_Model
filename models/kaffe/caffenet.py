from models.kaffe.network import Network

class CaffeNet(Network):
     def __init__(self, inputs, net_weights, trainable=True, use_dropout=0.0, keep_prob=0.5):
          self.keep_prob = keep_prob
          super(CaffeNet, self).__init__(inputs, net_weights, trainable, use_dropout)

     def setup(self):
        (self.feed('input')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .lrn(2, 2e-05, 0.75, name='norm1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .lrn(2, 2e-05, 0.75, name='norm2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
             .fc(4096, name='fc6')
             .dropout(self.keep_prob, name="drop6")
             .fc(4096, name='fc7')
             .dropout(self.keep_prob, name="drop7")
             .fc(1000, relu=False, name='fc8'))
        # should not have softmax here, unless the input size exactly matches
