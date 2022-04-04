import numpy as np
from utils.printer import Printer
import torch

class Evaluator(object):
    def __init__(self, num_class, printer, mode = 'train'):
        self.num_class = num_class
        self.printer = printer
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        # confusion_matrix中对角线的是预测正确的数量
        return np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc


    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        ''' 
            Real  False True
        Predict 
        False      0     1
        True       2     3
        '''
        count = np.bincount(label, minlength = self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        self.IoU_class = MIoU
        MIoU = np.nanmean(MIoU)
        return MIoU


    def add_batch(self, gt_image, pre_image):
        if gt_image.size()[1] == 1:
            gt_image.squeeze_(dim=1)
        if pre_image.size()[1] == 1:
            pre_image.squeeze_(dim=1)
        pre_image = pre_image.cpu().numpy()
        gt_image = gt_image.cpu().numpy()
        assert gt_image.shape == pre_image.shape, 'gt_image_shape: {} != pre_image: {}'.format(gt_image.shape, pre_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        del gt_image, pre_image

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
