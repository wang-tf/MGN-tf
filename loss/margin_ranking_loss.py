import tensorflow as tf
import tensorflow.keras as keras


class MarginRankingLoss(keras.losses.Loss):
    def __init__(self, margin, reduction='mean'):
        self.margin = margin
        self.reduction = reduction

    def call(self, x1, x2, y):
        if self.reduction != 'mean':
            raise
        return tf.math.reduce_mean(tf.math.maximum(0, -y * (x1 - x2) + self.margin))
