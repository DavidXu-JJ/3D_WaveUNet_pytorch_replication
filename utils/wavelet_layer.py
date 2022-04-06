import numpy as np
from utils.image_3D_io import load_image_3d
import math, pywt
from torch.nn import Module
from utils.wavelet_function import *

class DWT_1D(Module):
    """
    input: (N, C, L)
    output: L -- (N, C, L/2)
            H -- (N, C, L/2)
    """
    def __init__(self, wavename):
        super(DWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        print(self.band_low)
        self.band_high.reverse()
        print(self.band_high)
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L, L1 + self.band_length - 2) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2) )
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2

        matrix_h = matrix_h[:, (self.band_length_half-1):end]
        matrix_g = matrix_g[:, (self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.tensor(matrix_h).cuda()
            self.matrix_high = torch.tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.tensor(matrix_h)
            self.matrix_high = torch.tensor(matrix_g)

    def forward(self, input):
        assert len(input.size()) == 3
        self.input_height = input.size()[-1]
        self.get_matrix()
        return DWTFunction_1D.apply(input, self.matrix_low, self.matrix_high)

class IDWT_1D(Module):
    """
    input:  L -- (N, C, L/2)
            H -- (N, C, L/2)
    output: (N, C, L)
    """
    def __init__(self, wavename):
        super(IDWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:,(self.band_length_half-1):end]
        matrix_g = matrix_g[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.tensor(matrix_h).cuda()
            self.matrix_high = torch.tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.tensor(matrix_h)
            self.matrix_high = torch.tensor(matrix_g)

    def forward(self, L, H):
        assert len(L.size()) == len(H.size()) == 3
        self.input_height = L.size()[-1] + H.size()[-1]
        self.get_matrix()
        return IDWTFunction_1D.apply(L, H, self.matrix_low, self.matrix_high)

class DWT_2D(Module):
    """
    input: (N, C, H, W)
    output -- LL: (N, C, H/2, W/2)
              LH: (N, C, H/2, W/2)
              HL: (N, C, H/2, W/2)
              HH: (N, C, H/2, W/2)
    """
    def __init__(self, wavename):
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1/2)
        matrix_h = np.zeros( ( L, L1 + self.band_length - 2) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2

        # trim on the first dimension
        matrix_h_0 = matrix_h[0 : math.floor(self.input_height / 2), 0 : self.input_height + self.band_length - 2]
        matrix_h_1 = matrix_h[0 : math.floor(self.input_width / 2), 0 : self.input_width + self.band_length - 2]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2

        # trim on the first dimension
        matrix_g_0 = matrix_g[0 : self.input_height - math.floor(self.input_height / 2), 0 : self.input_height + self.band_length - 2]
        matrix_g_1 = matrix_g[0 : self.input_width - math.floor(self.input_width / 2), 0 : self.input_width + self.band_length - 2]

        # trim on the second dimension
        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1) : end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1) : end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1) : end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1) : end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.tensor(matrix_h_0)
            self.matrix_low_1 = torch.tensor(matrix_h_1)
            self.matrix_high_0 = torch.tensor(matrix_g_0)
            self.matrix_high_1 = torch.tensor(matrix_g_1)

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)

class DWT_2D_tiny(Module):
    def __init__(self, wavename):
        super(DWT_2D_tiny, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1/2)
        matrix_h = np.zeros( ( L, L1 + self.band_length - 2) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2

        # trim on the first dimension
        matrix_h_0 = matrix_h[0 : math.floor(self.input_height / 2), 0 : self.input_height + self.band_length - 2]
        matrix_h_1 = matrix_h[0 : math.floor(self.input_width / 2), 0 : self.input_width + self.band_length - 2]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2

        # trim on the first dimension
        matrix_g_0 = matrix_g[0 : self.input_height - math.floor(self.input_height / 2), 0 : self.input_height + self.band_length - 2]
        matrix_g_1 = matrix_g[0 : self.input_width - math.floor(self.input_width / 2), 0 : self.input_width + self.band_length - 2]

        # trim on the second dimension
        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1) : end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1) : end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1) : end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1) : end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.tensor(matrix_h_0)
            self.matrix_low_1 = torch.tensor(matrix_h_1)
            self.matrix_high_0 = torch.tensor(matrix_g_0)
            self.matrix_high_1 = torch.tensor(matrix_g_1)

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D_tiny.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)

class IDWT_2D(Module):
    """
    input -- LL: (N, C, H/2, W/2)
             LH: (N, C, H/2, W/2)
             HL: (N, C, H/2, W/2)
             HH: (N, C, H/2, W/2)
    output: (N, C, H, W)
    """
    def __init__(self, wavename):
        super(IDWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_low.reverse()
        self.band_high = wavelet.dec_hi
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1/2)
        matrix_h = np.zeros( ( L, L1 + self.band_length - 2) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2

        # trim on the first dimension
        matrix_h_0 = matrix_h[0 : math.floor(self.input_height / 2), 0 : self.input_height + self.band_length - 2]
        matrix_h_1 = matrix_h[0 : math.floor(self.input_width / 2), 0 : self.input_width + self.band_length - 2]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2

        # trim on the first dimension
        matrix_g_0 = matrix_g[0 : self.input_height - math.floor(self.input_height / 2), 0 : self.input_height + self.band_length - 2]
        matrix_g_1 = matrix_g[0 : self.input_width - math.floor(self.input_width / 2), 0 : self.input_width + self.band_length - 2]

        # trim on the second dimension
        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1) : end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1) : end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1) : end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1) : end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.tensor(matrix_h_0)
            self.matrix_low_1 = torch.tensor(matrix_h_1)
            self.matrix_high_0 = torch.tensor(matrix_g_0)
            self.matrix_high_1 = torch.tensor(matrix_g_1)

    def forward(self, LL, LH, HL, HH):
        assert(len(LL.size()) == len(LH.size()) == len(HL.size()) == len(HH.size()) == 4)
        self.input_height = LL.size()[-2] + HH.size()[-2]
        self.input_width = LL.size()[-1] + HH.size()[-1]
        self.get_matrix()
        return IDWTFunction_2D.apply(LL, LH, HL, HH, self.matrix_low_0, self.matrix_low-1, self.matrix_high_0, self.matrix_high_1)

class DWT_3D(Module):
    """
    input: (N, C, D, H, W)
    output: -- LLL (N, C, D/2, H/2, W/2)
            -- LLH (N, C, D/2, H/2, W/2)
            -- LHL (N, C, D/2, H/2, W/2)
            -- LHH (N, C, D/2, H/2, W/2)
            -- HLL (N, C, D/2, H/2, W/2)
            -- HLH (N, C, D/2, H/2, W/2)
            -- HHL (N, C, D/2, H/2, W/2)
            -- HHH (N, C, D/2, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波分解所用低频滤波器组
        :param band_high: 小波分解所用高频滤波器组
        """
        super(DWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        self.kernel_size = self.band_length
        self.stride = 2
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.Tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.Tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_low_2 = torch.Tensor(matrix_h_2)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)
            self.matrix_high_2 = torch.Tensor(matrix_g_2)

    def forward(self, input):
        assert len(input.size()) == 5
        self.input_depth = input.size()[-3]
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        #assert self.input_height > self.band_length and self.input_width > self.band_length and self.input_depth > self.band_length
        self.get_matrix()
        return DWTFunction_3D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                           self.matrix_high_0, self.matrix_high_1, self.matrix_high_2)


class DWT_3D_tiny(Module):
    """
    input: (N, C, D, H, W)
    output: -- LLL (N, C, D/2, H/2, W/2)
            -- LLH (N, C, D/2, H/2, W/2)
            -- LHL (N, C, D/2, H/2, W/2)
            -- LHH (N, C, D/2, H/2, W/2)
            -- HLL (N, C, D/2, H/2, W/2)
            -- HLH (N, C, D/2, H/2, W/2)
            -- HHL (N, C, D/2, H/2, W/2)
            -- HHH (N, C, D/2, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波分解所用低频滤波器组
        :param band_high: 小波分解所用高频滤波器组
        """
        super(DWT_3D_tiny, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        self.kernel_size = self.band_length
        self.stride = 2
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.Tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.Tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_low_2 = torch.Tensor(matrix_h_2)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)
            self.matrix_high_2 = torch.Tensor(matrix_g_2)

    def forward(self, input):
        assert len(input.size()) == 5
        self.input_depth = input.size()[-3]
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_3D_tiny.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                         self.matrix_high_0, self.matrix_high_1, self.matrix_high_2)


class IDWT_3D(Module):
    """
    input:  -- LLL (N, C, D/2, H/2, W/2)
            -- LLH (N, C, D/2, H/2, W/2)
            -- LHL (N, C, D/2, H/2, W/2)
            -- LHH (N, C, D/2, H/2, W/2)
            -- HLL (N, C, D/2, H/2, W/2)
            -- HLH (N, C, D/2, H/2, W/2)
            -- HHL (N, C, D/2, H/2, W/2)
            -- HHH (N, C, D/2, H/2, W/2)
    output: (N, C, D, H, W)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波重构所用低频滤波器组
        :param band_high: 小波重构所用高频滤波器组
        """
        super(IDWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        self.kernel_size = self.band_length
        self.stride = 2
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.Tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.Tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_low_2 = torch.Tensor(matrix_h_2)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)
            self.matrix_high_2 = torch.Tensor(matrix_g_2)

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        assert len(LLL.size()) == len(LLH.size()) == len(LHL.size()) == len(LHH.size()) == 5
        assert len(HLL.size()) == len(HLH.size()) == len(HHL.size()) == len(HHH.size()) == 5
        self.input_depth = LLL.size()[-3] + HHH.size()[-3]
        self.input_height = LLL.size()[-2] + HHH.size()[-2]
        self.input_width = LLL.size()[-1] + HHH.size()[-1]
        #assert self.input_height > self.band_length and self.input_width > self.band_length and self.input_depth > self.band_length
        self.get_matrix()
        return IDWTFunction_3D.apply(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
                                     self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                     self.matrix_high_0, self.matrix_high_1, self.matrix_high_2)


if __name__ == '__main__':
    a = DWT_2D("db4")
    input=np.zeros((4,4,7,16))
    input=torch.from_numpy(input)
    a.forward(input)
