import torch
import torch.nn as nn
import torch.nn.functional as F
from ..net_util import init_net
import numpy as np

class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=1, no_residual=True, last_op=None):
        super(SurfaceClassifier, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels
        self.last_op = last_op

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        '''

        :param feature: list of [BxC_inxHxW] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''

        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

        if self.last_op:
            y = self.last_op(y)

        return y

class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

class Fourier(nn.Module):
    def __init__(self, opt, nmb=16, scale=10):
        super(Fourier, self).__init__()

        device_ids = [int(i) for i in opt.gpu_ids.split(",")]
        cuda = torch.device('cuda:%d' % device_ids[0])

        self.nmb = nmb
        self.b = torch.randn(3, nmb, device=cuda)*scale
        self.pi = 3.14159265359
    def forward(self, v):
        batches, channels, num_samples = v.shape
        v = v.permute(0, 2, 1).reshape(batches * num_samples, channels)
        x_proj = torch.matmul(2*self.pi*v, self.b)
        x_proj = x_proj.view(batches,num_samples,self.nmb)
        x_proj = x_proj.permute(0,2,1)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], 1)

class SurfaceClassifierLinear(nn.Module):
    def __init__(self, opt, filter_channels, use_fourier = False, num_views=1, no_residual=True, last_op=None):
        super(SurfaceClassifierLinear, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels
        self.last_op = last_op
        self.fourier = Fourier(opt) if use_fourier else None

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Linear(
                    filter_channels[l],
                    filter_channels[l + 1]))
                self.add_module("linear%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Linear(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1]))
                else:
                    self.filters.append(nn.Linear(
                        filter_channels[l],
                        filter_channels[l + 1]))

                self.add_module("linear%d" % l, self.filters[l])

    def forward(self, feature):
        '''

        :param feature: list of [BxC_inxHxW] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''

        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['linear' + str(i)](y)
            else:
                y = self._modules['linear' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 2)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

        if self.last_op:
            y = self.last_op(y)

        return y