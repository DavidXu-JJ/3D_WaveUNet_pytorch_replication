import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

class CrossEntropy_Loss(Module):
    def __init__(self, w_crossentropy, class_weight):
        super(CrossEntropy_Loss, self).__init__()
        self.w_crossentropy = w_crossentropy
        self.class_weight = class_weight

    def forward(self, inputs, targets):
        loss_crossentropy = torch.nn.CrossEntropyLoss(size_average = True,
                                                      weight = self.class_weight).cuda()
        self.loss_crossentropy_value = loss_crossentropy.forward(input = inputs,
                                                                 target = targets)
        self.loss = self.w_crossentropy * self.loss_crossentropy_value
        return self.loss, self.__str__()

    def __str__(self):
        return '{} * {:.6f} = {:.6f}'.format(self.w_crossentropy,
                                             float(self.loss_crossentropy_value.data.cpu().numpy()),
                                             float(self.loss.data.cpu().numpy())
                                             )
