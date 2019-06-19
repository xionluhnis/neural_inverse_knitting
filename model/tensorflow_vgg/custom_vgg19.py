import os
import sys
import tensorflow as tf

import numpy as np
import time

# sys.path.insert(0, './')
from . import vgg19
from . import vgg16

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19(vgg19.Vgg19):
    # Input should be an gray image [batch, height, width, 1]
    # values scaled [-0.5, 0.5]
    def build(self, gray, train=False):
        start_time = time.time()
        print('build model started')

        net = dict()
        with tf.variable_scope('vgg19', reuse=tf.AUTO_REUSE):
            gray_scaled = (gray+0.5) * 255.0

            # Convert RGB to BGR
            # red, green, blue = tf.split(3, 3, rgb_scaled)
            bgr = tf.concat([
                gray_scaled - VGG_MEAN[0],
                gray_scaled - VGG_MEAN[1],
                gray_scaled - VGG_MEAN[2],
            ], axis=3)

            layer_names = ["conv1_1", "conv1_2", 'pool1',
                           "conv2_1", "conv2_2", 'pool2',
                           "conv3_1", "conv3_2", "conv3_3", "conv3_4", 'pool3',
                           "conv4_1", "conv4_2", "conv4_3", "conv4_4", 'pool4',
                           "conv5_1", "conv5_2", "conv5_3", "conv5_4", 'pool5']

            activation = bgr
            for layer in layer_names:
                if layer.startswith('conv'):
                    net[layer] = self.conv_layer(activation, layer)
                elif layer.startswith('pool'):
                    net[layer] = self.avg_pool(activation, layer)
                else:
                    raise('Error!')
                
                activation = net[layer]

            # self.conv1_1 = self.conv_layer(bgr, "conv1_1")
            # self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
            # self.pool1 = self.avg_pool(self.conv1_2, 'pool1')

            # self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
            # self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
            # self.pool2 = self.avg_pool(self.conv2_2, 'pool2')

            # self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
            # self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
            # self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
            # self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
            # self.pool3 = self.avg_pool(self.conv3_4, 'pool3')

            # self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
            # self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
            # self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
            # self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
            # self.pool4 = self.avg_pool(self.conv4_4, 'pool4')

            # self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
            # self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
            # self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
            # self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
            # self.pool5 = self.avg_pool(self.conv5_4, 'pool5')

        return net
        # self.data_dict = None
        print('build model finished: {}s'.format(time.time() - start_time))



class Vgg16(vgg16.Vgg16):
    # Input should be an gray image [batch, height, width, 1]
    # values scaled [-0.5, 0.5]
    def build(self, gray, train=False):
        start_time = time.time()
        print('build model started')

        net = dict()
        with tf.variable_scope('vgg16', reuse=tf.AUTO_REUSE):
            gray_scaled = (gray+0.5) * 255.0

            # Convert RGB to BGR
            # red, green, blue = tf.split(3, 3, rgb_scaled)
            bgr = tf.concat([
                gray_scaled - VGG_MEAN[0],
                gray_scaled - VGG_MEAN[1],
                gray_scaled - VGG_MEAN[2],
            ], axis=3)

            layer_names = ["conv1_1", "conv1_2", 'pool1',
                           "conv2_1", "conv2_2", 'pool2',
                           "conv3_1", "conv3_2", "conv3_3", 'pool3',
                           "conv4_1", "conv4_2", "conv4_3", 'pool4',
                           "conv5_1", "conv5_2", "conv5_3", 'pool5']

            activation = bgr
            for layer in layer_names:
                if layer.startswith('conv'):
                    net[layer] = self.conv_layer(activation, layer)
                elif layer.startswith('pool'):
                    net[layer] = self.avg_pool(activation, layer)
                else:
                    raise('Error!')
                
                activation = net[layer]

            # self.conv1_1 = self.conv_layer(bgr, "conv1_1")
            # self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
            # self.pool1 = self.avg_pool(self.conv1_2, 'pool1')

            # self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
            # self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
            # self.pool2 = self.avg_pool(self.conv2_2, 'pool2')

            # self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
            # self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
            # self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
            # self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
            # self.pool3 = self.avg_pool(self.conv3_4, 'pool3')

            # self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
            # self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
            # self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
            # self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
            # self.pool4 = self.avg_pool(self.conv4_4, 'pool4')

            # self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
            # self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
            # self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
            # self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
            # self.pool5 = self.avg_pool(self.conv5_4, 'pool5')

        return net
        # self.data_dict = None
        print('build model finished: {}s'.format(time.time() - start_time))
