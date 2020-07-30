#!/usr/bin/env python3
# coding:utf-8


import tensorflow.keras as keras


class Bottleneck(keras.Model):
    expansion = 4

    def __init__(self, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = keras.layers.Conv2D(planes, 1)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(planes, 3, strides=stride, padding='same')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2D(planes * self.expansion, 1)
        self.bn3 = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsamole is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
