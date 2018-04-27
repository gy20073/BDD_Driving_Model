from models.kaffe.network import Network

class CaffeNet_from_pool5(Network):
    def __init__(self, inputs, net_weights, trainable=True, use_dropout=0.0, keep_prob=0.5):
        # input should be previous tensor, should not be preprocessed
        # should provide weight file
        # Trainable = True
        # use_dropout = 1.0
        # keep_prob = 0.5
        self.keep_prob = keep_prob
        super(CaffeNet_from_pool5, self).__init__(inputs, net_weights, trainable, use_dropout)

    def setup(self):
        (self.feed('conv5')
        .max_pool(6, 6, 4, 4, padding='VALID', name='pool5_x2')
        .fc(4096, name='fc6')
        .dropout(self.keep_prob, name="drop6")
        .fc(4096, name='fc7')
        .dropout(self.keep_prob, name="drop7"))

        '''
        net=NET({"conv5": conv5},
                FLAGS.pretrained_model_path,
                use_dropout=use_dropout)
        '''