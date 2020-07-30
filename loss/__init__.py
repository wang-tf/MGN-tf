import os
import numpy as np
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

# from loss.triplet import TripletLoss, TripletSemihardLoss
import tensorflow_addons as tfa
from tensorflow_addons.losses import TripletHardLoss as TripletLoss


class MGNLoss(keras.losses.Loss):
    def __init__(self, args, ckpt):
        super().__init__()
        print('[INFO] Making loss...')

        self.nGPU = args.nGPU
        self.args = args
        self.loss = []
        self.loss_module = list()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'CrossEntropy':
                loss_function = keras.losses.CategoricalCrossentropy()
            elif loss_type == 'Triplet':
                loss_function = TripletLoss(args.margin)

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
                })
            

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        # self.log = torch.Tensor()

        # device = torch.device('cpu' if args.cpu else 'cuda')
        # self.loss_module.to(device)
        
        if args.load != '': self.load(ckpt.dir, cpu=args.cpu)
        # if not args.cpu and args.nGPU > 1:
        #     self.loss_module = nn.DataParallel(self.loss_module, range(args.nGPU))

    def call(self, labels, outputs):
        losses = []
        for i, l in enumerate(self.loss):
            if self.args.model == 'MGN' and l['type'] == 'Triplet':
                loss = [l['function'](labels, output) for output in outputs[1:4]]
                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                # self.log[-1, i] += effective_loss.item()
            elif self.args.model == 'MGN' and l['function'] is not None:
                loss = []
                lf = l['function']
                one_hot_labels = tf.keras.utils.to_categorical(labels, self.args.num_classes)
                loss = [l['function'](one_hot_labels, output) for output in outputs[4:]]
                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                # self.log[-1, i] += effective_loss.item()
            else:
                pass
        loss_sum = sum(losses)
        # if len(self.loss) > 1:
        #     self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, batches):
        self.log[-1].div_(batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.jpg'.format(apath, l['type']))
            plt.close(fig)

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def get_loss_module(self):
        if self.nGPU == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

