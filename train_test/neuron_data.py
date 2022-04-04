from utils.image_3D_io import save_iamge_3d, load_image_3d
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
from PIL import Image,ImageOps, ImageFilter
from constant import Mean_TrainData, Std_TrainData, Block_Size

class Neuron_Dataset(Dataset):
    def __init__(self,
                 root,
                 source,
                 channel = 1,
                 depth = 32,
                 height = 128,
                 width = 128,
                 model = 'train'):
        self.root = root
        self.source = source
        self.neuron_name_list = self._neuron_name_list()
        self.channel = channel
        self.depth = depth
        self.height = height
        self.width = width
        self.num_class = 2
        self.model = model

    def _neuron_name_list(self):
        """
        生成完整的神经元数据路径名列表
        :return:
        """
        assert os.path.isfile(self.source)
        neuron_name_list = open(self.source).readlines()
        neuron_name_list = [line.strip() for line in neuron_name_list]
        neuron_name_list = [line for line in neuron_name_list if not line.startswith('#')]
        return [line for line in neuron_name_list if os.path.isdir(os.path.join(self.root, line))]


    def __getitem__(self, item):
        data_path = os.path.join(self.root, self.neuron_name_list[item])
        image = load_image_3d(image_root = os.path.join(data_path, 'image'))
        label = load_image_3d(image_root = os.path.join(data_path, 'label'))
        sample = {'image': image, 'label': label}
        if self.model == 'train':
            return self.transform_tr(sample)
        elif self.model == 'test' or 'val':
            return self.transform_val(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            RandomFlip(direction = 'x', p = 0.5),
            RandomFlip(direction='y', p = 0.5),
            RandomNoise(p = 0.5, sigma = 1, mean = 0),
            Normalize(mean = Mean_TrainData, std = Std_TrainData),
            ToTensor()
        ])
        return composed_transforms(sample)
    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            Normalize(mean=Mean_TrainData, std=Std_TrainData),
            ToTensor()
        ])
        return composed_transforms(sample)

    def __len__(self):
        return len(self.neuron_name_list)

    def __repr__(self):
        text = 'A neuron DATA SET generated from "{}"'.format(self.root)
        return text

class Normalize():
    def __init__(self, mean = 0.0, std = 1.):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        image = sample['image']
        image = np.array(image).astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        sample['image'] = image
        return sample

class ToTensor():
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image = np.array(image, dtype = np.float32)
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        sample['image'] = image
        sample['label'] = label
        return sample

class RandomNoise():
    def __init__(self, p = 0.5, sigma = 1., mean = 0.):
        self.p = p
        self.sigma = sigma
        self.mean = mean

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        noise = self.sigma * np.random.rand(*(image.shape)) + self.mean
        image = image + noise
        sample['image'] = image
        sample['label'] = label
        return sample

class RandomFlip():
    def __init__(self, direction = 'x', p = 0.5):
        self.p = p
        self.direction = direction.lower()
        assert self.direction in ['x', 'y']

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random()  < self.p:
            if self.direction == 'x':
                image = image[:,:,::-1].copy()
                label = label[:,:,::-1].copy()
            if self.direction == 'y':
                image = image[:, ::-1, :].copy()
                label = label[:, ::-1, :].copy()
        sample['image'] = image
        sample['label'] = label
        return sample
