
import torch, os

NUM_CLASSES = 2

# 神经元图像数据是以二维图像序列形式保存的，其后缀名是一致的
ImageSuffixes = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']

Image2DName_Length = 6
Image3DName_Length = 4

Block_Size = (32, 128, 128)     #(z, y, x) -- depth, height, width

# DataTrain_Root = '/Users/davidxu/NeuCuDa/DataBase_5_new'
DataTrain_Root = 'G:\\NeuCuDa\\DataBase_5_new'
#训练数据保存路径，其中保存的是 neuronal cubes
TrainSource = os.path.join(DataTrain_Root, 'train.txt') #训练数据
TestSource = os.path.join(DataTrain_Root, 'test.txt')   #测试数据

Mean_TrainData = 0.029720289547802054   # 这是基于 BigNeuron 数据生成的图像块的均值和方差
Std_TrainData = 0.04219472495471814


NeuronNet = {'neuron_segnet', 'neuron_unet_v1', 'neuron_unet_v2',
             'neuron_wavesnet_v1', 'neuron_wavesnet_v2', 'neuron_wavesnet_v3', 'neuron_wavesnet_v4'}

# DataNeuron_Root = '/Users/davidxu/NeuCuDa/DataBase_1_resized'
DataNeuron_Root = 'G:\\NeuCuDa\\DataBase_1_resized'
