import cv2
import collections
import random
from data.common import list_pictures


class Market1501():
    def __init__(self, args, transform, dtype):

        self.transform = transform

        data_path = args.datadir
        if dtype == 'train':
            data_path += '/bounding_box_train'
        elif dtype == 'test':
            data_path += '/bounding_box_test'
        else:
            data_path += '/query'

        
        self.imgs = [path for path in list_pictures(data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

        self._id2index = collections.defaultdict(list)
        if dtype == 'train':
            self.sampled_imgs = self.randomsampler(args.batchid, args.batchimage)

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = cv2.imread(path)
        img = img[::-1]
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __call__(self):
        for index in self.sampled_imgs:
            
            path = self.imgs[index]
            target = self._id2label[self.id(path)]
            img = cv2.imread(path)
            img = img[::-1]
            
            if self.transform is not None:
                img = self.transform(img)

            yield img, target
            
    def __item__(self):
        img, target = self.__getitem__(self.index)
        self.index += 1
        return img, target

    def __len__(self):
        return len(self.imgs)

    def randomsampler(self, batch_id, batch_image):
        self.batch_image = batch_image
        self.batch_id = batch_id

        for idx, path in enumerate(self.imgs):
            _id = self.id(path)
            self._id2index[_id].append(idx)

        random.shuffle(self.unique_ids)

        sampled_imgs = []
        for _id in self.unique_ids:
            sampled_imgs.extend(self._sample(self._id2index[_id], self.batch_image))
        return sampled_imgs

    @staticmethod
    def _sample(population, k):
        if len(population) < k:
            population = population * k
        return random.sample(population, k)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]
