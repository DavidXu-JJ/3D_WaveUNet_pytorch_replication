import torch
import os, math
from datetime import datetime
from torch.nn import *
from train_test.neuron_data import Neuron_Dataset
from train_test.networks import *
from train_test.neuron_loss import CrossEntropy_Loss
from torch.utils.data import DataLoader
from torch import optim
from train_test.neuron_label_compare import Evaluator


class Train_Test_Process():
    def __init__(self, args):
        self.args = args
        self.printer = self.args.printer
        self._data()
        self.net = self._get_net()
        self.loss = CrossEntropy_Loss(w_crossentropy = 1,
                                      class_weight = self.args.class_weight)
        self.optimizer = self._optizimer()
        print(self.args.cuda)
        if self.args.cuda:
            self.device = torch.device("cuda", 0)
            # self.net = DataParallel(module = self.net.cuda(),
            #                         device_ids = self.args.gpu,
            #                         output_device = self.args.out_gpu)
            self.net = self.net.to(self.device)
            self.loss = self.loss.to(self.device)

        self.best_pred = 0.0

        # for key, value in self.args.__dict__.items():
        #     if not key.startswith('_'):
        #         self.printer.pprint('{} ==> {}'.format(key.rjust(24), value))


    def _get_net(self):
        if self.args.net == 'neuron_segnet':
            return Neuron_SegNet(num_class = self.num_class)
        elif self.args.net == 'neuron_unet_v1':
            return Neuron_UNet_V1(num_class = self.num_class)
        elif self.args.net == 'neuron_unet_v2':
            return Neuron_UNet_V2(num_class = self.num_class)
        elif self.args.net == 'neuron_wavesnet_v1':
            return Neuron_WaveSNet_V1(num_class = self.num_class)
        elif self.args.net == 'neuron_wavesnet_v2':
            return Neuron_WaveSNet_V2(num_class = self.num_class)
        elif self.args.net == 'neuron_wavesnet_v3':
            return Neuron_WaveSNet_V3(num_class = self.num_class)

    def _data(self):
        self.train_root = self.args.dataroot
        self.test_root = self.args.dataroot
        self.neuron_train_set = Neuron_Dataset(root = self.train_root,
                                               source = self.args.data_train_source,
                                               depth = self.args.depth,
                                               height = self.args.height,
                                               width = self.args.width)
        self.iters_per_epoch = len(self.neuron_train_set)
        self.train_data_loader = DataLoader(dataset = self.neuron_train_set,
                                            batch_size = self.args.batch_size,
                                            shuffle = True,
                                            num_workers = self.args.workers)
        self.neuron_test_set = Neuron_Dataset(root = self.test_root,
                                                source = self.args.data_test_source,
                                                depth=self.args.depth,
                                                height=self.args.height,
                                                width=self.args.width)
        self.test_data_loader = DataLoader(dataset = self.neuron_test_set,
                                           batch_size=self.args.batch_size,
                                           shuffle=True,
                                           num_workers=self.args.workers)
        self.num_class = self.neuron_train_set.num_class
        self.evaluator_train = Evaluator(num_class = self.num_class,
                                         printer = self.args.printer,
                                         mode = 'train')
        self.evaluator_test = Evaluator(num_class = self.num_class,
                                        printer = self.args.printer,
                                        mode = 'test')


    def _optizimer(self):
        weight_p, bias_p = [], []
        for name, p in self.net.named_parameters():
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        return optim.SGD([
            {'params': weight_p, 'weight_decay': self.args.weight_decay},
            {'params': bias_p, 'weight_dacay': 0}],
            lr = self.args.lr,
            momentum = self.args.momentum)


    def adjust_learning_rate(self, epoch, iteration):
        T = epoch * self.iters_per_epoch + iteration
        N = self.args.epochs * self.iters_per_epoch
        warmup_iters = self.args.warmup_epochs * self.iters_per_epoch

        if self.args.lr_scheduler == 'step':
            lr = self.args.lr * (0.1 ** (epoch // self.args.lr_step))
        elif self.args.lr_scheduler == 'poly':
            lr = self.args.lr * pow((1 - 1.0 * T / N), 0.9)
        elif self.args.lr_scheduler == 'cos':
            lr = 0.5 * self.args.lr * (1 + math.cos(1.0 * T / N * math.pi))
        else:
            raise ValueError(
                'lr_scheduler is {}, which should be in ["step", "poly", "cos"]'.format(self.args.lr_scheduler))

        if warmup_iters > 0 and T < warmup_iters:
            lr = lr * 1.0 * T / warmup_iters

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def test(self):
        with torch.no_grad():
            for test_loader in [self.test_data_loader,]:
                start = datetime.now()
                self.evaluator_test.reset()
                loss_total = 0.0
                self.net.eval()
                length = len(test_loader)
                for index, sample in enumerate(test_loader):
                    start_1 = datetime.now()
                    test_image = sample['image']
                    test_label = sample['label']

                    test_image.unsqueeze_(dim=1)

                    if self.args.cuda:
                        test_image = test_image.to(device=self.device, dtype=torch.float32)
                        test_label = test_label.to(device=self.device, dtype=torch.long)

                    output = self.net.forward(test_image)
                    test_loss, str_loss = self.loss.forward(inputs = output,
                                                            targets = test_label)
                    loss_total += test_loss.data
                    _, predict_label = output.topk(1, dim=1)

                    self.evaluator_test.add_batch(gt_image = test_label,
                                                  pre_image = predict_label)
                    stop_1 = datetime.now()
                    acc = self.evaluator_test.Pixel_Accuracy()
                    mIoU = self.evaluator_test.Mean_Intersection_over_Union()

                    text = '{:4d}/{}, train_loss = {:.6f} / {}, acc = {:.6f}, mIoU = {:.6f}, took {}.'.format(
                        index, length,
                        test_loss, loss_total / (index + 1),
                        acc, mIoU, stop_1 - start_1
                    )
                    self.printer.pprint('testing - ' + text)

                self.printer.pprint('testing totally ---- ')

                mIoU = self.evaluator_test.Mean_Intersection_over_Union()
                if mIoU > self.best_pred:
                    self.best_pred = mIoU
                    filename = os.path.join(self.args.weight_root, self.args.checkname + '_best.pth.tar')
                    torch.save({'state_dict': self.net.state_dict(),
                                'best_pred': self.best_pred,}, filename)

                stop = datetime.now()
                text = 'testing took {}.'.format(stop-start)
                self.printer.pprint(text = text)
                self.printer.pprint(' ')

        self.net.train()

    def train(self):
        start = datetime.now()
        number = 0
        train_length = len(self.train_data_loader)
        for epoch in range(self.args.epochs):
            start_0 = datetime.now()
            self.epoch = epoch
            train_loss_epoch = 0.0
            self.evaluator_train.reset()
            for iteration_in_epoch, sample in enumerate(self.train_data_loader):
                start_1 = datetime.now()
                train_image = sample['image']
                train_label = sample['label']

                train_image.unsqueeze_(dim=1)

                if self.args.cuda:
                    train_image = train_image.to(device=self.device, dtype=torch.float32)
                    train_label = train_label.to(device=self.device, dtype=torch.long)


                output = self.net.forward(train_image)
                train_loss, str_loss = self.loss.forward(inputs = output,
                                                         targets = train_label)

                train_loss_epoch += train_loss
                _, predict_label = output.topk(1, dim = 1)
                self.evaluator_train.add_batch(gt_image = train_label,
                                               pre_image = predict_label)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                stop_1 = datetime.now()

                acc = self.evaluator_train.Pixel_Accuracy()
                mIoU = self.evaluator_train.Mean_Intersection_over_Union()

                text = 'Epoch {:3d}/{:3d}, batch {:4d}/{}, train_loss = {:.6f} / {}, acc = {:.6f}, mIoU = {:.6f}, took {} / {}'.format(
                    epoch, self.args.epochs, iteration_in_epoch, train_length,
                    train_loss, train_loss_epoch / (iteration_in_epoch + 1),
                    acc, mIoU, stop_1 - start_1, stop_1 - start
                )

                self.printer.pprint('training - ' + text)

                number += 1
                self.adjust_learning_rate(epoch = epoch, iteration = iteration_in_epoch)

            self.test()
            if (epoch + 1) % self.args.epoch_to_save == 0 or (epoch + 1) == self.args.epochs:
                text = 'saving weights ...'
                self.printer.pprint(text)
                filename = os.path.join(self.args.weight_root, 'epoch_{}'.format(epoch) + '.pth.tar')
                torch.save({'epoch': epoch + 1, 'state_dict': self.net.state_dict(),
                            'optimizer': self.optimizer.state_dict(), 'best_pred': self.best_pred,}, filename)

            stop_0 = datetime.now()
            text = 'Epoch - {:3d}, train_loss = {:.6f}, took {} hours -- {}'.format(epoch,
                                                                                    train_loss_epoch / self.iters_per_epoch,
                                                                                    stop_0 - start_0,
                                                                                    stop_0 - start)
            self.printer.pprint(text)
            self.printer.pprint(' ')
        stop = datetime.now()
        text = 'train_test_process finish, took {} hours !!'.format(stop - start)
        self.printer.pprint(text)




