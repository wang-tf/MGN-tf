import os
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc as misc

import tensorflow.keras as keras
from tensorflow.keras import optimizers as optim


class checkpoint():
    def __init__(self, args):
        self.args = args
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '':
            if args.save == '': args.save = now
            self.dir = 'experiment/' + args.save
        else:
            self.dir = 'experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = ''
            else:
                self.log = torch.load(self.dir + '/map_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)*args.test_every))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_map_rank(epoch)
        torch.save(self.log, os.path.join(self.dir, 'map_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False, end='\n'):
        print(log, end=end)
        if end != '':
            self.log_file.write(log + end)
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_map_rank(self, epoch):
        axis = np.linspace(1, epoch, self.log.size(0))
        label = 'Reid on {}'.format(self.args.data_test)
        labels = ['mAP','rank1','rank3','rank5','rank10']
        fig = plt.figure()
        plt.title(label)
        for i in range(len(labels)):
            plt.plot(axis, self.log[:, i].numpy(), label=labels[i])

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('mAP/rank')
        plt.grid(True)
        plt.savefig('{}/test_{}.jpg'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        pass

def make_optimizer(args, model):
    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {
            'momentum': args.momentum,
            'dampening': args.dampening,
            'nesterov': args.nesterov
            }
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'beta_1': args.beta1,
            'beta_2': args.beta2,
            'epsilon': args.epsilon,
            'amsgrad': args.amsgrad
        }
    elif args.optimizer == 'NADAM':
        optimizer_function = optim.Nadam
        kwargs = {
            'bata_1': args.beta1,
            'bata_2': args.beta2,
            'epsilon': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {
            'epsilon': args.epsilon,
            'momentum': args.momentum
        }
    else:
        raise Exception()

    kwargs['lr'] = args.lr
    kwargs['decay'] = args.weight_decay
    
    return optimizer_function(**kwargs)

def make_scheduler(args, optimizer):
    if args.decay_type == 'step':
        scheduler = keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=args.lr,
                decay_steps=args.lr_decay,
                decay_rate=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = keras.optimizers.schedules.PiecewiseConstantDecay(
                milestones,
                [args.lr * (args.gamma ** i) for i in range(len(milestones)+1)] 
        )

    return scheduler

