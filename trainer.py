#!/usr/bin/env python3
# coding:utf-8


import os
import tensorflow as tf
import numpy as np
import utils.utility as utility
from scipy.spatial.distance import cdist
from utils.functions import cmc, mean_ap
from utils.re_ranking import re_ranking


class Trainer():
    def __init__(self, args, model, loss, loader, ckpt):
        self.args = args
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.query_loader = loader.query_loader
        self.testset = loader.testset
        self.queryset = loader.queryset

        self.ckpt = ckpt
        self.model = model
        self.loss = loss
        self.lr = 0.
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if args.nGPU > 1:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            self.distributed_train_loader = self.mirrored_strategy.experimental_distribute_dataset(self.train_loader)

        if args.load != '':
            self.optimizer.load_state_dict(
                # torch.load(os.path.join(ckpt.dir, 'optimizer.pt'))
                keras.models.load_model(os.path.join(ckpt.dir, 'optimizer.h5'))
            )
            for _ in range(len(ckpt.log)*args.test_every): self.scheduler.step()

    @staticmethod
    def train_step(inputs, model, loss, optimizer):
        features, labels = inputs
        with tf.GradientTape() as tape:
            outputs = model(features)
            total_loss = loss.call(labels, outputs)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return total_loss
    
    def train(self):
        if self.args.nGPU > 1:
            for batch, inputs in enumerate(self.distributed_train_loader):
                print(f'batch: {batch}')
                total_loss = self.distributed_train_step(inputs, self.model, self.loss, self.optimizer)
                print(f'total loss: {total_loss}')
        else:
            for batch, inputs in enumerate(self.train_loader):
                print(f'batch: {batch}')
                total_loss = self.train_step(inputs, self.model, self.loss, self.optimizer)
                print(f'total loss: {total_loss}')

    def distributed_train_step(self, dist_inputs, model, loss, optimizer):
        per_replica_losses = self.mirrored_strategy.run(self.train_step, args=(dist_inputs, model, loss, optimizer))
        return self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    def train2(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model.fit(self.train_loader, epochs=self.args.epochs)

        raise
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        if lr != self.lr:
            self.ckpt.write_log('[INFO] Epoch: {}\tLearning rate: {:.2e}'.format(epoch, lr))
            self.lr = lr
        self.loss.start_log()
        self.model.train()

        for batch, (inputs, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                epoch, self.args.epochs,
                batch + 1, len(self.train_loader),
                self.loss.display_loss(batch)), 
            end='' if batch+1 != len(self.train_loader) else '\n')

        self.loss.end_log(len(self.train_loader))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckpt.write_log('\n[INFO] Test:')
        self.model.eval()

        self.ckpt.add_log(tf.zeros((1, 5)))
        qf = self.extract_feature(self.query_loader).numpy()
        gf = self.extract_feature(self.test_loader).numpy()

        if self.args.re_rank:
            q_g_dist = np.dot(qf, np.transpose(gf))
            q_q_dist = np.dot(qf, np.transpose(qf))
            g_g_dist = np.dot(gf, np.transpose(gf))
            dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        else:
            dist = cdist(qf, gf)
        r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True)
        m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

        self.ckpt.log[-1, 0] = m_ap
        self.ckpt.log[-1, 1] = r[0]
        self.ckpt.log[-1, 2] = r[2]
        self.ckpt.log[-1, 3] = r[4]
        self.ckpt.log[-1, 4] = r[9]
        best = self.ckpt.log.max(0)
        self.ckpt.write_log(
            '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} (Best: {:.4f} @epoch {})'.format(
            m_ap,
            r[0], r[2], r[4], r[9],
            best[0][0],
            (best[1][0] + 1)*self.args.test_every
            )
        )
        if not self.args.test_only:
            self.ckpt.save(self, epoch, is_best=((best[1][0] + 1)*self.args.test_every == epoch))

    def fliphor(self, inputs):
        inv_idx = keras.backend.arange(inputs.shape[3]-1, -1, -1).as_type(tf.long)  # N x C x H x W
        return inputs.index_select(3, inv_idx)

    def extract_feature(self, loader):
        for (inputs, labels) in loader:
            ff = tf.zeros((inputs.shape[0], 2048))
            for i in range(2):
                if i==1:
                    inputs = tf.image.flip_left_right(inputs)
                outputs = self.model(inputs)
                f = outputs[0].data.cpu()
                ff = ff + f

            # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            fnorm = tf.norm(ff, ord=2, axis=1, keepdims=True)
            ff = ff.div(fnorm.expand_as(ff))

            features = torch.concat((features, ff), 0)
        return features

