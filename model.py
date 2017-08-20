import tensorflow as tf
import numpy as np
import sys
from network import *

class Model:
    @staticmethod 
    def alexnet(net_input, keep_rate):
        # TODO weight decay loss tern
        # Layer 1 (conv-relu-pool-lrn)
        conv1 = conv(net_input, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        conv1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        # Layer 2 (conv-relu-pool-lrn)
        conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
        conv2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        # Layer 3 (conv-relu)
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
        # Layer 4 (conv-relu)
        conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
        # Layer 5 (conv-relu-pool)
        conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        # Layer 6 (fc-relu-drop)
        fc6 = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(fc6, 6*6*256, 4096, name='fc6')
        fc6 = dropout(fc6, keep_rate)
        # Layer 7 (fc-relu-drop)
        fc7 = fc(fc6, 4096, 4096, name='fc7')
        fc7 = dropout(fc7, keep_rate)
        # Layer 8 (fc-prob)
        fc8 = fc(fc7, 4096, 20, relu=False, name='fc8')
        return fc8
        
