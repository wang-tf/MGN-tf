#!/usr/bin/env python3
# coding:utf-8

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.applications import ResNet50 as resnet50
from .bottleneck import Bottleneck
from .res_conv4 import ResConv4
from .res_g_conv5 import ResGConv5


def make_model(args):
    return MGN(args)


class MGN(keras.Model):
    def __init__(self, args):
        super(MGN, self).__init__()
        num_classes = args.num_classes

        resnet = resnet50(input_shape=(None, None, 3), include_top=False)
        resnet_layer_names = [layer.name for layer in resnet.layers]

        conv4_block1_out = resnet.get_layer(name='conv4_block1_out')
        self.backone = keras.Model(inputs=resnet.inputs, outputs=conv4_block1_out.output)

        res_conv4 = ResConv4(batch_input_shape=(None, None, None, 1024))
        res_conv4.build((None, None, None, 1024))
        res_conv4.init(resnet)

        res_g_conv5 = ResGConv5(strides=2)
        res_g_conv5.build((None, None, None, 1024))
        res_g_conv5.init(resnet)

        res_p_conv5 = ResGConv5(strides=1)
        res_p_conv5.build((None, None, None, 1024))
        res_p_conv5.init(resnet)

        self.p1 = keras.Sequential([res_conv4, res_g_conv5])
        self.p2 = keras.Sequential([res_conv4, res_p_conv5])
        self.p3 = keras.Sequential([res_conv4, res_p_conv5])
        
        if args.pool == 'max':
            pool2d = keras.layers.MaxPool2D
        elif args.pool == 'avg':
            pool2d = keras.layers.AvgPool2D
        else:
            raise Exception()

        self.maxpool_zg_p1 = pool2d(pool_size=(12, 4))
        self.maxpool_zg_p2 = pool2d(pool_size=(24, 8))
        self.maxpool_zg_p3 = pool2d(pool_size=(24, 8))
        self.maxpool_zp2 = pool2d(pool_size=(12, 8))
        self.maxpool_zp3 = pool2d(pool_size=(8, 8))

        he_init = keras.initializers.VarianceScaling(scale=2., mode='fan_in', distribution='truncated_normal')
        norm_init = tf.random_normal_initializer(mean=1., stddev=0.02)
        reduction = keras.Sequential([keras.layers.Conv2D(args.feats, 1, kernel_initializer=he_init, use_bias=False), keras.layers.BatchNormalization(gamma_initializer=norm_init), keras.layers.ReLU()])

        # self._init_reduction(reduction)
        self.reduction_0 = keras.models.clone_model(reduction)
        self.reduction_1 = keras.models.clone_model(reduction)
        self.reduction_2 = keras.models.clone_model(reduction)
        self.reduction_3 = keras.models.clone_model(reduction)
        self.reduction_4 = keras.models.clone_model(reduction)
        self.reduction_5 = keras.models.clone_model(reduction)
        self.reduction_6 = keras.models.clone_model(reduction)
        self.reduction_7 = keras.models.clone_model(reduction)

        he_init_fan_out = keras.initializers.VarianceScaling(scale=2., mode='fan_out', distribution='truncated_normal')
        #self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_0 = keras.layers.Dense(num_classes, kernel_initializer=he_init_fan_out)
        self.fc_id_2048_1 = keras.layers.Dense(num_classes, kernel_initializer=he_init_fan_out)
        self.fc_id_2048_2 = keras.layers.Dense(num_classes, kernel_initializer=he_init_fan_out)

        self.fc_id_256_1_0 = keras.layers.Dense(num_classes, kernel_initializer=he_init_fan_out)
        self.fc_id_256_1_1 = keras.layers.Dense(num_classes, kernel_initializer=he_init_fan_out)
        self.fc_id_256_2_0 = keras.layers.Dense(num_classes, kernel_initializer=he_init_fan_out)
        self.fc_id_256_2_1 = keras.layers.Dense(num_classes, kernel_initializer=he_init_fan_out)
        self.fc_id_256_2_2 = keras.layers.Dense(num_classes, kernel_initializer=he_init_fan_out)

    def call(self, x):

        x = self.backone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        # z0_p2 = zp2[:, :, 0:1, :]
        z0_p2 = zp2[:, 0:1, :, :]
        # z1_p2 = zp2[:, :, 1:2, :]
        z1_p2 = zp2[:, 1:2, :, :]

        zp3 = self.maxpool_zp3(p3)
        # z0_p3 = zp3[:, :, 0:1, :]
        z0_p3 = zp3[:, 0:1, :, :]
        # z1_p3 = zp3[:, :, 1:2, :]
        z1_p3 = zp3[:, 1:2, :, :]
        # z2_p3 = zp3[:, :, 2:3, :]
        z2_p3 = zp3[:, 2:3, :, :]
        
        fg_p1 = tf.squeeze(tf.squeeze(self.reduction_0(zg_p1), axis=2), axis=1)
        fg_p2 = tf.squeeze(tf.squeeze(self.reduction_1(zg_p2), axis=2), axis=1)
        fg_p3 = tf.squeeze(tf.squeeze(self.reduction_2(zg_p3), axis=2), axis=1)
        f0_p2 = tf.squeeze(tf.squeeze(self.reduction_3(z0_p2), axis=2), axis=1)
        f1_p2 = tf.squeeze(tf.squeeze(self.reduction_4(z1_p2), axis=2), axis=1)
        f0_p3 = tf.squeeze(tf.squeeze(self.reduction_5(z0_p3), axis=2), axis=1)
        f1_p3 = tf.squeeze(tf.squeeze(self.reduction_6(z1_p3), axis=2), axis=1)
        f2_p3 = tf.squeeze(tf.squeeze(self.reduction_7(z2_p3), axis=2), axis=1)

        '''
        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
        '''
        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)
        
        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        predict = tf.concat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], axis=1)

        return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3

