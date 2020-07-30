#!/usr/bin/env python3
# coding:utf-8 


import tensorflow.keras as keras

from .resnetblock import ResNetBlock


class ResConv4(keras.Model):
    def __init__(self, name_scope='', batch_input_shape=None):
        super().__init__()
        batch_input_shape = (None, None, None, 1024)
        self.conv4_block2 = ResNetBlock(batch_input_shape, name_scope='conv4_block2')
        self.conv4_block3 = ResNetBlock(batch_input_shape, name_scope='conv4_block3')
        self.conv4_block4 = ResNetBlock(batch_input_shape, name_scope='conv4_block4')
        self.conv4_block5 = ResNetBlock(batch_input_shape, name_scope='conv4_block5')
        self.conv4_block6 = ResNetBlock(batch_input_shape, name_scope='conv4_block6')

    def init(self, resnet):
        self.conv4_block2.init(resnet)
        self.conv4_block3.init(resnet)
        self.conv4_block4.init(resnet)
        self.conv4_block5.init(resnet)
        self.conv4_block6.init(resnet)

    def call(self, inputs):
        out = inputs

        out = self.conv4_block2(out)
        out = self.conv4_block3(out)
        out = self.conv4_block4(out)
        out = self.conv4_block5(out)
        out = self.conv4_block6(out)
        return out

