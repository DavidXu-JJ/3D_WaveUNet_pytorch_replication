import torch
from torch import nn
from torch.nn import Module
from datetime import datetime
import torch.nn.functional as F


class Conv3x3_BN(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 dilation = 1,
                 groups = 1,
                 bias = True,
                 padding_mode = 'zeros',
                 with_BN = True):
        super(Conv3x3_BN, self).__init__()
        self.with_BN = with_BN
        self.conv = nn.Conv3d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding,
                              dilation = dilation,
                              groups = groups,
                              bias = bias)

        self.bn = nn.BatchNorm3d(num_features = out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        if self.with_BN:
            return self.relu(self.bn(self.conv(input)))
        else:
            return self.relu(self.conv(input))

# use Maxpooling to downsample, use maxunpooling to upsample
class Neuron_SegNet(Module):
    def __init__(self, num_class = 2, with_BN = True, channel_width = 4):
        super(Neuron_SegNet, self).__init__()
        # Batch * 1 * 32 * 128 * 128 => Batch * channel_width * 32 * 128 * 128
        self.conv3d_11_en = Conv3x3_BN(in_channels=1,
                                       out_channels=1*channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * channel_width * 32 * 128 * 128 => Batch * channel_width * 32 * 128 * 128
        # same
        self.conv3d_12_en = Conv3x3_BN(in_channels=1*channel_width,
                                       out_channels=1*channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * channel_width * 32 * 128 * 128 => Batch * channel_width * 16 * 64 * 64
        self.downsampling_1 = nn.MaxPool3d(kernel_size=2,
                                           return_indices=True)

        # Batch * channel_width * 16 * 64 * 64 => Batch * 2*channel_width * 16 * 64 * 64
        self.conv3d_21_en = Conv3x3_BN(in_channels=1*channel_width,
                                       out_channels=2*channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * 2*channel_width * 16 * 64 * 64 => Batch * 2*channel_width * 16 * 64 * 64
        # same
        self.conv3d_22_en = Conv3x3_BN(in_channels=2*channel_width,
                                       out_channels=2*channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * 2*channel_width * 16 * 64 * 64 => Batch * 2*channel_width * 8 * 32 * 32
        self.downsampling_2 = nn.MaxPool3d(kernel_size=2,
                                           return_indices=True)

        # Batch * 2*channel_width * 8 * 32 * 32 => Batch * 4*channel_width * 8 * 32 * 32
        self.conv3d_31_en = Conv3x3_BN(in_channels=2*channel_width,
                                       out_channels=4*channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * 4*channel_width * 8 * 32 * 32 => Batch * 4*channel_width * 8 * 32 * 32
        # same
        self.conv3d_32_en = Conv3x3_BN(in_channels=4*channel_width,
                                       out_channels=4*channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * 4*channel_width * 8 * 32 * 32 => Batch * 4*channel_width * 4 * 16 * 16
        self.downsampling_3 = nn.MaxPool3d(kernel_size=2,
                                           return_indices=True)

       # Batch * 4*channel_width * 4 * 16 * 16 => Batch * 8*channel_width * 4 * 16 * 16
        self.conv3d_41_en = Conv3x3_BN(in_channels=4*channel_width,
                                       out_channels=8*channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * 8*channel_width * 4 * 16 * 16 => Batch * 8*channel_width * 4 * 16 * 16
        # same
        self.conv3d_42_en = Conv3x3_BN(in_channels=8*channel_width,
                                       out_channels=8*channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * 8*channel_width * 4 * 16 * 16 => Batch * 8*channel_width * 2 * 8 * 8
        self.downsampling_4 = nn.MaxPool3d(kernel_size=2,
                                           return_indices=True)

        # Batch * 8*channel_width * 2 * 8 * 8 => Batch * 8*channel_width * 2 * 8 * 8
        # same
        self.conv3d_51 = Conv3x3_BN(in_channels=8*channel_width,
                                    out_channels=8*channel_width,
                                    kernel_size=3,
                                    padding=1,
                                    with_BN=with_BN)
        self.conv3d_52 = Conv3x3_BN(in_channels=8*channel_width,
                                    out_channels=8*channel_width,
                                    kernel_size=3,
                                    padding=1,
                                    with_BN=with_BN)

        # Batch * 8*channel_width * 2 * 8 * 8 => Batch * 8*channel_width * 4 * 16 * 16
        self.upsampling_4 = nn.MaxUnpool3d(kernel_size = 2,
                                           stride = 2)

        # Batch * 8*channel_width * 4 * 16 * 16 => Batch * 8*channel_width * 4 * 16 * 16
        # same
        self.conv3d_41_de = Conv3x3_BN(in_channels=8*channel_width,
                                       out_channels=8*channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * 8*channel_width * 4 * 16 * 16 => Batch * 4*channel_width * 4 * 16 * 16
        # same
        self.conv3d_42_de = Conv3x3_BN(in_channels=8*channel_width,
                                       out_channels=4*channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 4*channel_width * 4 * 16 * 16 => Batch * 4*channel_width * 8 * 32 * 32
        self.upsampling_3 = nn.MaxUnpool3d(kernel_size=2,
                                           stride=2)

        # Batch * 4*channel_width * 8 * 32 * 32 => Batch * 4*channel_width * 8 * 32 * 32
        # same
        self.conv3d_31_de = Conv3x3_BN(in_channels=4*channel_width,
                                       out_channels=4*channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 4*channel_width * 8 * 32 * 32 => Batch * 2*channel_width * 8 * 32 * 32
        self.conv3d_32_de = Conv3x3_BN(in_channels=4*channel_width,
                                       out_channels=2*channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 2*channel_width * 8 * 32 * 32 => Batch * 2*channel_width * 16 * 64 * 64
        self.upsampling_2 = nn.MaxUnpool3d(kernel_size=2,
                                           stride=2)

        # Batch * 2*channel_width * 16 * 64 * 64 => Batch * 2*channel_width * 16 * 64 * 64
        # same
        self.conv3d_21_de = Conv3x3_BN(in_channels=2*channel_width,
                                       out_channels=2*channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 2*channel_width * 16 * 64 * 64 => Batch * channel_width * 16 * 64 * 64
        self.conv3d_22_de = Conv3x3_BN(in_channels=2*channel_width,
                                       out_channels=channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * channel_width * 16 * 64 * 64 => Batch * channel_width * 32 * 128 * 128
        self.upsampling_1 = nn.MaxUnpool3d(kernel_size=2,
                                           stride=2)
        # Batch * channel_width * 32 * 128 * 128 => Batch * channel_width * 32 * 128 * 128
        # same
        self.conv3d_11_de = Conv3x3_BN(in_channels=channel_width,
                                       out_channels=channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * channel_width * 32 * 128 * 128 => Batch * channel_width * 32 * 128 * 128
        # same
        self.conv3d_12_de = Conv3x3_BN(in_channels=channel_width,
                                       out_channels=channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        self.conv_final = nn.Conv3d(in_channels=channel_width,
                                    out_channels=num_class,
                                    kernel_size=1)

    def forward(self, input):
        output = self.conv3d_12_en(self.conv3d_11_en(input))
        output, indices_1 = self.downsampling_1(output)

        output = self.conv3d_22_en(self.conv3d_21_en(output))
        output, indices_2 = self.downsampling_2(output)

        output = self.conv3d_32_en(self.conv3d_31_en(output))
        output, indices_3 = self.downsampling_3(output)

        output = self.conv3d_42_en(self.conv3d_41_en(output))
        output, indices_4 = self.downsampling_4(output)

        output = self.conv3d_52(self.conv3d_51(output))

        output = self.upsampling_4(input = output, indices = indices_4)
        output = self.conv3d_42_de(self.conv3d_41_de(output))

        output = self.upsampling_3(input = output, indices = indices_3)
        output = self.conv3d_32_de(self.conv3d_31_de(output))


        output = self.upsampling_2(input = output, indices=indices_2)
        output = self.conv3d_22_de(self.conv3d_21_de(output))

        output = self.upsampling_1(input = output, indices = indices_1)
        output = self.conv3d_12_de(self.conv3d_11_de(output))

        output = self.conv_final(output)

        return output

# use maxpooling to downsample, use convtranspose to upsample
class Neuron_UNet_V1(Module):
    def __init__(self, num_class=2, with_BN=True, channel_width=4):
        super(Neuron_UNet_V1, self).__init__()
        # Batch * 1 * 32 * 128 * 128 => Batch * channel_width * 32 * 128 * 128
        self.conv3d_11_en = Conv3x3_BN(in_channels=1,
                                       out_channels=1 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * channel_width * 32 * 128 * 128 => Batch * channel_width * 32 * 128 * 128
        # same
        self.conv3d_12_en = Conv3x3_BN(in_channels=1 * channel_width,
                                       out_channels=1 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * channel_width * 32 * 128 * 128 => Batch * channel_width * 16 * 64 * 64
        self.downsampling_1 = nn.MaxPool3d(kernel_size=2)

        # Batch * channel_width * 16 * 64 * 64 => Batch * 2*channel_width * 16 * 64 * 64
        self.conv3d_21_en = Conv3x3_BN(in_channels=1 * channel_width,
                                       out_channels=2 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * 2*channel_width * 16 * 64 * 64 => Batch * 2*channel_width * 16 * 64 * 64
        # same
        self.conv3d_22_en = Conv3x3_BN(in_channels=2 * channel_width,
                                       out_channels=2 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * 2*channel_width * 16 * 64 * 64 => Batch * 2*channel_width * 8 * 32 * 32
        self.downsampling_2 = nn.MaxPool3d(kernel_size=2)

        # Batch * 2*channel_width * 8 * 32 * 32 => Batch * 4*channel_width * 8 * 32 * 32
        self.conv3d_31_en = Conv3x3_BN(in_channels=2 * channel_width,
                                       out_channels=4 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * 4*channel_width * 8 * 32 * 32 => Batch * 4*channel_width * 8 * 32 * 32
        # same
        self.conv3d_32_en = Conv3x3_BN(in_channels=4 * channel_width,
                                       out_channels=4 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * 4*channel_width * 8 * 32 * 32 => Batch * 4*channel_width * 4 * 16 * 16
        self.downsampling_3 = nn.MaxPool3d(kernel_size=2)

        # Batch * 4*channel_width * 4 * 16 * 16 => Batch * 8*channel_width * 4 * 16 * 16
        self.conv3d_41_en = Conv3x3_BN(in_channels=4 * channel_width,
                                       out_channels=8 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * 8*channel_width * 4 * 16 * 16 => Batch * 8*channel_width * 4 * 16 * 16
        # same
        self.conv3d_42_en = Conv3x3_BN(in_channels=8 * channel_width,
                                       out_channels=8 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * 8*channel_width * 4 * 16 * 16 => Batch * 8*channel_width * 2 * 8 * 8
        self.downsampling_4 = nn.MaxPool3d(kernel_size=2)

        # Batch * 8*channel_width * 2 * 8 * 8 => Batch * 8*channel_width * 2 * 8 * 8
        # same
        self.conv3d_51 = Conv3x3_BN(in_channels=8 * channel_width,
                                    out_channels=8 * channel_width,
                                    kernel_size=3,
                                    padding=1,
                                    with_BN=with_BN)
        self.conv3d_52 = Conv3x3_BN(in_channels=8 * channel_width,
                                    out_channels=8 * channel_width,
                                    kernel_size=3,
                                    padding=1,
                                    with_BN=with_BN)

        # Batch * 8*channel_width * 2 * 8 * 8 => Batch * 8*channel_width * 4 * 16 * 16
        self.upsampling_4 = nn.ConvTranspose3d(in_channels=8*channel_width,
                                               out_channels=8*channel_width,
                                               kernel_size=1,
                                               stride=2,
                                               output_padding=1,
                                               padding=0)

        # Batch * 16*channel_width * 4 * 16 * 16 => Batch * 8*channel_width * 4 * 16 * 16
        # has encode information
        self.conv3d_41_de = Conv3x3_BN(in_channels=16 * channel_width,
                                       out_channels=8 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * 8*channel_width * 4 * 16 * 16 => Batch * 4*channel_width * 4 * 16 * 16
        self.conv3d_42_de = Conv3x3_BN(in_channels=8 * channel_width,
                                       out_channels=4 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 4*channel_width * 4 * 16 * 16 => Batch * 4*channel_width * 8 * 32 * 32
        self.upsampling_3 = nn.ConvTranspose3d(in_channels = 4 * channel_width,
                                               out_channels = 4 * channel_width,
                                               kernel_size = 1,
                                               stride = 2,
                                               output_padding = 1,
                                               padding = 0)

        # Batch * 8*channel_width * 8 * 32 * 32 => Batch * 4*channel_width * 8 * 32 * 32
        # has encode information
        self.conv3d_31_de = Conv3x3_BN(in_channels=8 * channel_width,
                                       out_channels=4 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 4*channel_width * 8 * 32 * 32 => Batch * 2*channel_width * 8 * 32 * 32
        self.conv3d_32_de = Conv3x3_BN(in_channels=4 * channel_width,
                                       out_channels=2 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 2*channel_width * 8 * 32 * 32 => Batch * 2*channel_width * 16 * 64 * 64
        self.upsampling_2 = nn.ConvTranspose3d(in_channels = 2 * channel_width,
                                               out_channels = 2 * channel_width,
                                               kernel_size = 1,
                                               stride = 2,
                                               output_padding = 1,
                                               padding = 0)

        # Batch * 4*channel_width * 16 * 64 * 64 => Batch * 2*channel_width * 16 * 64 * 64
        # has encode information
        self.conv3d_21_de = Conv3x3_BN(in_channels=4 * channel_width,
                                       out_channels=2 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 2*channel_width * 16 * 64 * 64 => Batch * channel_width * 16 * 64 * 64
        self.conv3d_22_de = Conv3x3_BN(in_channels=2 * channel_width,
                                       out_channels=channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * channel_width * 16 * 64 * 64 => Batch * channel_width * 32 * 128 * 128
        self.upsampling_1 = nn.ConvTranspose3d(in_channels = 1 * channel_width,
                                               out_channels = 1 * channel_width,
                                               kernel_size = 1,
                                               stride = 2,
                                               output_padding = 1,
                                               padding = 0)
        # Batch * 2*channel_width * 32 * 128 * 128 => Batch * channel_width * 32 * 128 * 128
        # has encode information
        self.conv3d_11_de = Conv3x3_BN(in_channels=2 * channel_width,
                                       out_channels=channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * channel_width * 32 * 128 * 128 => Batch * channel_width * 32 * 128 * 128
        # same
        self.conv3d_12_de = Conv3x3_BN(in_channels=channel_width,
                                       out_channels=channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        self.conv_final = nn.Conv3d(in_channels=channel_width,
                                    out_channels=num_class,
                                    kernel_size=1)


    def forward(self, input):
        output_1 = self.conv3d_12_en(self.conv3d_11_en(input))
        output = self.downsampling_1(output_1)

        output_2 = self.conv3d_22_en(self.conv3d_21_en(output))
        output = self.downsampling_2(output_2)

        output_3 = self.conv3d_32_en(self.conv3d_31_en(output))
        output = self.downsampling_3(output_3)

        output_4 = self.conv3d_42_en(self.conv3d_41_en(output))
        output = self.downsampling_4(output_4)

        output = self.conv3d_52(self.conv3d_51(output))

        output = self.upsampling_4(input=output)
        output = self.conv3d_42_de(self.conv3d_41_de(torch.cat((output, output_4), dim=1)))

        output = self.upsampling_3(input=output)
        output = self.conv3d_32_de(self.conv3d_31_de(torch.cat((output, output_3), dim=1)))

        output = self.upsampling_2(input=output)
        output = self.conv3d_22_de(self.conv3d_21_de(torch.cat((output, output_2), dim=1)))

        output = self.upsampling_1(input=output)
        output = self.conv3d_12_de(self.conv3d_11_de(torch.cat((output, output_1), dim=1)))

        output = self.conv_final(output)

        return output

# use conv to downsample, use interpolate to upsample
class Neuron_UNet_V2(Module):
    def __init__(self, num_class=2, with_BN=True, channel_width=4):
        super(Neuron_UNet_V2, self).__init__()
        # Batch * 1 * 32 * 128 * 128 => Batch * channel_width * 32 * 128 * 128
        self.conv3d_11_en = Conv3x3_BN(in_channels=1,
                                       out_channels=1 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * channel_width * 32 * 128 * 128 => Batch * channel_width * 32 * 128 * 128
        # same
        self.conv3d_12_en = Conv3x3_BN(in_channels=1 * channel_width,
                                       out_channels=1 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * channel_width * 32 * 128 * 128 => Batch * channel_width * 16 * 64 * 64
        self.downsampling_1 = nn.Conv3d(in_channels=1 * channel_width,
                                        out_channels=1 * channel_width,
                                        kernel_size=1,
                                        stride=2)

        # Batch * channel_width * 16 * 64 * 64 => Batch * 2*channel_width * 16 * 64 * 64
        self.conv3d_21_en = Conv3x3_BN(in_channels=1 * channel_width,
                                       out_channels=2 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 2*channel_width * 16 * 64 * 64 => Batch * 2*channel_width * 16 * 64 * 64
        # same
        self.conv3d_22_en = Conv3x3_BN(in_channels=2 * channel_width,
                                       out_channels=2 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 2*channel_width * 16 * 64 * 64 => Batch * 2*channel_width * 8 * 32 * 32
        self.downsampling_2 = nn.Conv3d(in_channels=2 * channel_width,
                                        out_channels=2 * channel_width,
                                        kernel_size=1,
                                        stride=2)

        # Batch * 2*channel_width * 8 * 32 * 32 => Batch * 4*channel_width * 8 * 32 * 32
        self.conv3d_31_en = Conv3x3_BN(in_channels=2 * channel_width,
                                       out_channels=4 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 4*channel_width * 8 * 32 * 32 => Batch * 4*channel_width * 8 * 32 * 32
        # same
        self.conv3d_32_en = Conv3x3_BN(in_channels=4 * channel_width,
                                       out_channels=4 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 4*channel_width * 8 * 32 * 32 => Batch * 4*channel_width * 4 * 16 * 16
        self.downsampling_3 = nn.Conv3d(in_channels=4 * channel_width,
                                        out_channels=4 * channel_width,
                                        kernel_size=1,
                                        stride=2)

        # Batch * 4*channel_width * 4 * 16 * 16 => Batch * 8*channel_width * 4 * 16 * 16
        self.conv3d_41_en = Conv3x3_BN(in_channels=4 * channel_width,
                                       out_channels=8 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 8*channel_width * 4 * 16 * 16 => Batch * 8*channel_width * 4 * 16 * 16
        # same
        self.conv3d_42_en = Conv3x3_BN(in_channels=8 * channel_width,
                                       out_channels=8 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 8*channel_width * 4 * 16 * 16 => Batch * 8*channel_width * 2 * 8 * 8
        self.downsampling_4 = nn.Conv3d(in_channels=8 * channel_width,
                                        out_channels=8 * channel_width,
                                        kernel_size=1,
                                        stride=2)

        # Batch * 8*channel_width * 2 * 8 * 8 => Batch * 8*channel_width * 2 * 8 * 8
        # same
        self.conv3d_51 = Conv3x3_BN(in_channels=8 * channel_width,
                                    out_channels=8 * channel_width,
                                    kernel_size=3,
                                    padding=1,
                                    with_BN=with_BN)
        self.conv3d_52 = Conv3x3_BN(in_channels=8 * channel_width,
                                    out_channels=8 * channel_width,
                                    kernel_size=3,
                                    padding=1,
                                    with_BN=with_BN)

        # Batch * 8*channel_width * 2 * 8 * 8 => Batch * 8*channel_width * 4 * 16 * 16
        # upsample4

        # Batch * 16*channel_width * 4 * 16 * 16 => Batch * 8*channel_width * 4 * 16 * 16
        # has encode information
        self.conv3d_41_de = Conv3x3_BN(in_channels=16 * channel_width,
                                       out_channels=8 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)
        # Batch * 8*channel_width * 4 * 16 * 16 => Batch * 4*channel_width * 4 * 16 * 16
        self.conv3d_42_de = Conv3x3_BN(in_channels=8 * channel_width,
                                       out_channels=4 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 4*channel_width * 4 * 16 * 16 => Batch * 4*channel_width * 8 * 32 * 32
        # upsample3

        # Batch * 8*channel_width * 8 * 32 * 32 => Batch * 4*channel_width * 8 * 32 * 32
        # has encode information
        self.conv3d_31_de = Conv3x3_BN(in_channels=8 * channel_width,
                                       out_channels=4 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 4*channel_width * 8 * 32 * 32 => Batch * 2*channel_width * 8 * 32 * 32
        self.conv3d_32_de = Conv3x3_BN(in_channels=4 * channel_width,
                                       out_channels=2 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 2*channel_width * 8 * 32 * 32 => Batch * 2*channel_width * 16 * 64 * 64
        # upsample2

        # Batch * 4*channel_width * 16 * 64 * 64 => Batch * 2*channel_width * 16 * 64 * 64
        # has encode information
        self.conv3d_21_de = Conv3x3_BN(in_channels=4 * channel_width,
                                       out_channels=2 * channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * 2*channel_width * 16 * 64 * 64 => Batch * channel_width * 16 * 64 * 64
        self.conv3d_22_de = Conv3x3_BN(in_channels=2 * channel_width,
                                       out_channels=channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * channel_width * 16 * 64 * 64 => Batch * channel_width * 32 * 128 * 128
        # upsample1

        # Batch * 2*channel_width * 32 * 128 * 128 => Batch * channel_width * 32 * 128 * 128
        # has encode information
        self.conv3d_11_de = Conv3x3_BN(in_channels=2 * channel_width,
                                       out_channels=channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        # Batch * channel_width * 32 * 128 * 128 => Batch * channel_width * 32 * 128 * 128
        # same
        self.conv3d_12_de = Conv3x3_BN(in_channels=channel_width,
                                       out_channels=channel_width,
                                       kernel_size=3,
                                       padding=1,
                                       with_BN=with_BN)

        self.conv_final = nn.Conv3d(in_channels=channel_width,
                                    out_channels=num_class,
                                    kernel_size=1)


    def forward(self, input):
        output_1 = self.conv3d_12_en(self.conv3d_11_en(input))
        output = self.downsampling_1(output_1)

        output_2 = self.conv3d_22_en(self.conv3d_21_en(output))
        output = self.downsampling_2(output_2)

        output_3 = self.conv3d_32_en(self.conv3d_31_en(output))
        output = self.downsampling_3(output_3)

        output_4 = self.conv3d_42_en(self.conv3d_41_en(output))
        output = self.downsampling_4(output_4)

        output = self.conv3d_52(self.conv3d_51(output))

        B, C, D, H, W = output.size()
        output = F.interpolate(output, size = (2 * D, 2 * H, 2 * W), mode = 'trilinear', align_corners=True)
        output = self.conv3d_42_de(self.conv3d_41_de(torch.cat((output, output_4), dim=1)))

        B, C, D, H, W = output.size()
        output = F.interpolate(output, size = (2 * D, 2 * H, 2 * W), mode = 'trilinear', align_corners=True)
        output = self.conv3d_32_de(self.conv3d_31_de(torch.cat((output, output_3), dim=1)))

        B, C, D, H, W = output.size()
        output = F.interpolate(output, size = (2 * D, 2 * H, 2 * W), mode = 'trilinear', align_corners=True)
        output = self.conv3d_22_de(self.conv3d_21_de(torch.cat((output, output_2), dim=1)))

        B, C, D, H, W = output.size()
        output = F.interpolate(output, size = (2 * D, 2 * H, 2 * W), mode = 'trilinear', align_corners=True)
        output = self.conv3d_12_de(self.conv3d_11_de(torch.cat((output, output_1), dim=1)))

        output = self.conv_final(output)

        return output
