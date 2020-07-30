#!/usr/bin/env python3
# coding:utf-8

import tensorflow.keras as keras

from .resnetblock import ResNetBlock


class ResGConv5(keras.Model):
    def __init__(self, strides=2):
        super().__init__()
        self.conv5_block1 = ResNetBlock(batch_input_shape=(None, None, None, 1024), input_filters=512, output_filters=2048, name_scope='conv5_block1', strides=strides)
        self.conv5_block2 = ResNetBlock(batch_input_shape=(None, None, None, 2048), input_filters=512, output_filters=2048, name_scope='conv5_block2')
        self.conv5_block3 = ResNetBlock(batch_input_shape=(None, None, None, 2048), input_filters=512, output_filters=2048, name_scope='conv5_block3')

    def init(self, resnet):
        self.conv5_block1.init(resnet)
        self.conv5_block2.init(resnet)
        self.conv5_block3.init(resnet)

    def call(self, x):
        out = x

        out = self.conv5_block1(out)
        out = self.conv5_block2(out)
        out = self.conv5_block3(out)
        return out

