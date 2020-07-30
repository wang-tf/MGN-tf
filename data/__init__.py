import tensorflow as tf
from importlib import import_module

from .sampler import RandomSampler
from utils.random_erasing import RandomErasing


def normalize_func(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    normed_image = (image - mean) / std
    return normed_image


class Transform():
    def __init__(self, args):
        self.args = args
        self.random_erasing = RandomErasing(probability=args.probability, mean=[0.0, 0.0, 0.0])

    def train_transform(self, image):
        image = tf.image.resize(image, [self.args.height, self.args.width], method='bicubic')
        image = tf.image.random_flip_left_right(image)
        image = normalize_func(image)
        if self.args.random_erasing:
            image = self.random_erasing(image)
        return image

    def test_transform(self, image):
        image = tf.image.resize(image, [self.args.height, self.args.width], method='bicubic')
        image = normalize_func(image)
        return image


class Data():
    def __init__(self, args):
        transforms = Transform(args)
        train_transform = transforms.train_transform
        test_transform = transforms.test_transform

        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            self.trainset = getattr(module_train, args.data_train)(args, train_transform, 'train')
            self.train_loader = tf.data.Dataset.from_generator(self.trainset, (tf.float32, tf.int64)).batch(args.batchid * args.batchimage).take(args.batchid*args.batchimage)
        else:
            self.train_loader = None

        if args.data_test in ['Market1501']:
            module = import_module('data.' + args.data_train.lower())
            self.testset = getattr(module, args.data_test)(args, test_transform, 'test')
            self.queryset = getattr(module, args.data_test)(args, test_transform, 'query')

        else:
            raise Exception()

        self.test_loader = tf.data.Dataset.from_generator(lambda: (item for item in self.testset), (tf.float32, tf.int64)).batch(args.batchtest)
        self.query_loader = tf.data.Dataset.from_generator(lambda: (item for item in self.queryset), (tf.float32, tf.int64)).batch(args.batchtest)


class Data_backup:
    def __init__(self, args):

        train_list = [
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if args.random_erasing:
            train_list.append(RandomErasing(probability=args.probability, mean=[0.0, 0.0, 0.0]))

        train_transform = transforms.Compose(train_list)

        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            self.trainset = getattr(module_train, args.data_train)(args, train_transform, 'train')
            self.train_loader = dataloader.DataLoader(self.trainset,
                            sampler=RandomSampler(self.trainset,args.batchid,batch_image=args.batchimage),
                            #shuffle=True,
                            batch_size=args.batchid * args.batchimage,
                            num_workers=args.nThread)
        else:
            self.train_loader = None
        
        if args.data_test in ['Market1501']:
            module = import_module('data.' + args.data_train.lower())
            self.testset = getattr(module, args.data_test)(args, test_transform, 'test')
            self.queryset = getattr(module, args.data_test)(args, test_transform, 'query')

        else:
            raise Exception()

        self.test_loader = dataloader.DataLoader(self.testset, batch_size=args.batchtest, num_workers=args.nThread)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=args.batchtest, num_workers=args.nThread)
        
