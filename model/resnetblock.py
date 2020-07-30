#!/usr/bin/env python3
# coding:utf-8 


import tensorflow.keras as keras


class ResNetBlock(keras.Model):
    def __init__(self, batch_input_shape, input_filters=256, output_filters=1024, name_scope='', strides=1):
        super().__init__()
        self.batch_input_shape = batch_input_shape
        self.output_filters = output_filters

        if batch_input_shape[3] != output_filters:
            self.block_0_conv = keras.layers.Conv2D(output_filters, 1, strides=strides, padding='valid', name=f'{name_scope}_0_conv', batch_input_shape=batch_input_shape)
            self.block_0_bn = keras.layers.BatchNormalization(momentum=0.99, epsilon=1.001e-05, name=f'{name_scope}_0_bn')

        self.block_1_conv = keras.layers.Conv2D(input_filters, 1, strides=strides, padding='valid', name=f'{name_scope}_1_conv', batch_input_shape=batch_input_shape)
        self.block_1_bn = keras.layers.BatchNormalization(momentum=0.99, epsilon=1.001e-05, name=f'{name_scope}_1_bn')
        self.block_1_relu = keras.layers.ReLU(name=f'{name_scope}_1_relu')

        self.block_2_conv = keras.layers.Conv2D(input_filters, 3, strides=1, padding='same', name=f'{name_scope}_2_conv')
        self.block_2_bn = keras.layers.BatchNormalization(momentum=0.99, epsilon=1.001e-05, name=f'{name_scope}_2_bn')
        self.block_2_relu = keras.layers.ReLU(name=f'{name_scope}_2_relu')

        self.block_3_conv = keras.layers.Conv2D(output_filters, 1, strides=1, padding='valid', name=f'{name_scope}_3_conv')
        self.block_3_bn = keras.layers.BatchNormalization(momentum=0.99, epsilon=1.001e-05, name=f'{name_scope}_3_bn')

        # block_add = block_0_bn + block_3_bn
        self.block_out = keras.layers.ReLU(name=f'{name_scope}_out')

    def init(self, resnet):
        self.block_1_conv.set_weights(resnet.get_layer(self.block_1_conv.name).get_weights())
        self.block_1_bn.set_weights(resnet.get_layer(self.block_1_bn.name).get_weights())
        self.block_1_relu.set_weights(resnet.get_layer(self.block_1_relu.name).get_weights())

        self.block_2_conv.set_weights(resnet.get_layer(self.block_2_conv.name).get_weights())
        self.block_2_bn.set_weights(resnet.get_layer(self.block_2_bn.name).get_weights())
        self.block_2_relu.set_weights(resnet.get_layer(self.block_2_relu.name).get_weights())

        self.block_3_conv.set_weights(resnet.get_layer(self.block_3_conv.name).get_weights())
        self.block_3_bn.set_weights(resnet.get_layer(self.block_3_bn.name).get_weights())

    def call(self, x):
        out = x

        if self.batch_input_shape[3] != self.output_filters:
            x = self.block_0_conv(x)
            x = self.block_0_bn(x)

        out = self.block_1_conv(out)
        out = self.block_1_bn(out)
        out = self.block_1_relu(out)

        out = self.block_2_conv(out)
        out = self.block_2_bn(out)
        out = self.block_2_relu(out)

        out = self.block_3_conv(out)
        out = self.block_3_bn(out)

        out = x + out

        out = self.block_out(out)
        return out

