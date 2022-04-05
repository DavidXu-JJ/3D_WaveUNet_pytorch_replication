import argparse, os
import torch
import numpy as np
from datetime import datetime
from train_test.neuron_train_test import Train_Test_Process
from constant import NeuronNet, DataTrain_Root, TrainSource, TestSource
from utils.printer import Printer

def my_mkdir(file_name, mode = 'file'):
    """
    创建根路径
    :param mode: 'path', 'file'
    """
    if mode == 'path':
        if not os.path.isdir(file_name):
            os.makedirs(file_name)
            return
    elif mode == 'file':
        root, name = os.path.split(file_name)
        if not os.path.isdir(root):
            os.makedirs(root)
        return
    else:
        assert mode in ['path', 'file']


def main():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--net',type = str,default='neuron_unet_v2', choices = NeuronNet)
    parser.add_argument('--dataroot', type = str, default=DataTrain_Root, help = 'the path to the dataset')
    parser.add_argument('--data_train_source', type = str, default=TrainSource, help = 'the train image name list of the dataset')
    parser.add_argument('--data_test_source', type=str, default=TestSource, help='the test image name list of the dataset')
    parser.add_argument('--workers', type=int, default=1,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--data_size', type = str, default = '32,128,128',
                        help = 'the size of the input neuronal cube, must be a comma-separated list of three integers (default = 32,128,128)')
    parser.add_argument('--class_weight', type = str, default = '1.,5.',
                        help = 'the weights for the catergories in the loss function, must be a comma-separated list of three integers (default = 1,10.)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        metavar='N', help='the epochs for warming up the training')
    parser.add_argument('--epoch_to_save', type=int, default=1, metavar='N',
                        help='number of epochs to save the trained parameters')
    parser.add_argument('--batch_size', type=int, default=32,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--with_BN', action='store_true', default=True,
                        help='whether to use BN in the network model (default: True)')
    parser.add_argument('--lr', '--learning_ratio', type = float, default = 0.1, metavar = 'LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'], help='lr scheduler mode: (default: poly)')
    parser.add_argument('--lr_step', type=int, default=15,
                        metavar='N', help='epoch step to decay lr')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--no_cuda', action='store_true', default = False, help = 'disables CUDA training')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--checkname', type=str, default='test', help='set the checkpoint name')



    args = parser.parse_args()  #在执行这条命令之前，所有命令行参数都给不会生效
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            args.gpu_map = {}
            for index, gpu_id in enumerate(args.gpu_ids):
                args.gpu_map['cuda:{}'.format(gpu_id)] = 'cuda:{}'.format(index)
            args.gpu = list([i for i in range(len(args.gpu_ids))])
            args.out_gpu = 0
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    args.data_size = tuple(int(s) for s in args.data_size.split(','))
    args.depth = args.data_size[0]
    args.height = args.data_size[1]
    args.width = args.data_size[2]
    args.class_weight = torch.tensor(np.array(tuple(float(s) for s in args.class_weight.split(',')))).float()

    args.info_file = os.path.join('.', 'info', args.net, args.checkname + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.info'))
    args.weight_root = os.path.join('.', 'weight', args.net, args.checkname + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    my_mkdir(args.info_file, mode = 'file')
    my_mkdir(args.weight_root, mode = 'path')
    args.printer = Printer(args.info_file)

    ttper = Train_Test_Process(args)
    ttper.args.printer.pprint('Total Epoches: {}'.format(ttper.args.epochs))
    ttper.train()

if __name__ == '__main__':
    main()


