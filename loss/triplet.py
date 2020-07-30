#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow.keras as keras
from torch.nn import functional as F
from .margin_ranking_loss import MarginRankingLoss


class TripletSemihardLoss(keras.losses.Loss):
    """
    Shape:
        - Input: :math:`(N, C)` where `C = number of channels`
        - Target: :math:`(N)`
        - Output: scalar.
    """

    def __init__(self, device, margin=0, size_average=True):
        super(TripletSemihardLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average
        self.device = device

    def forward(self, input, target):
        y_true = target.int().unsqueeze(-1)
        same_id = torch.eq(y_true, y_true.t()).type_as(input)

        pos_mask = same_id
        neg_mask = 1 - same_id

        def _mask_max(input_tensor, mask, axis=None, keepdims=False):
            input_tensor = input_tensor - 1e6 * (1 - mask)
            _max, _idx = torch.max(input_tensor, dim=axis, keepdim=keepdims)
            return _max, _idx

        def _mask_min(input_tensor, mask, axis=None, keepdims=False):
            input_tensor = input_tensor + 1e6 * (1 - mask)
            _min, _idx = torch.min(input_tensor, dim=axis, keepdim=keepdims)
            return _min, _idx

        # output[i, j] = || feature[i, :] - feature[j, :] ||_2
        dist_squared = torch.sum(input ** 2, dim=1, keepdim=True) + \
                       torch.sum(input.t() ** 2, dim=0, keepdim=True) - \
                       2.0 * torch.matmul(input, input.t())
        dist = dist_squared.clamp(min=1e-16).sqrt()

        pos_max, pos_idx = _mask_max(dist, pos_mask, axis=-1)
        neg_min, neg_idx = _mask_min(dist, neg_mask, axis=-1)

        # loss(x, y) = max(0, -y * (x1 - x2) + margin)
        y = torch.ones(same_id.size()[0]).to(self.device)
        return F.margin_ranking_loss(neg_min.float(),
                                     pos_max.float(),
                                     y,
                                     self.margin,
                                     self.size_average)


class TripletLoss(keras.losses.Loss):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def call(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.shape[0]
        #inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist_sum = tf.keras.backend.sum(tf.math.pow(inputs, 2), axis=1, keepdims=True)
        dist = tf.broadcast_to(dist_sum, [n, n])
        dist = dist + tf.transpose(dist)
        dist += -2 * tf.linalg.matmul(inputs, tf.transpose(inputs))
        MAX = 100
        dist = tf.sqrt(tf.clip_by_value(dist, 1e-12, MAX))
        # For each anchor, find the hardest positive and negative
        mask = tf.equal(tf.broadcast_to(targets, [n, n]), tf.transpose(tf.broadcast_to(targets, [n, n])))
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(tf.expand_dims(tf.maximum(dist[i][mask[i]]), 0))
            dist_an.append(tf.expand_dims(tf.minimum(dist[i][mask[i] == 0]), 0))
        dist_ap = tf.concat(dist_ap)
        dist_an = tf.concat(dist_an)
        # Compute ranking hinge loss
        y = tf.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss
