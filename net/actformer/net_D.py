import math
import os.path as path

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from utils.skeleton.nn import GraphConv, RelativeQuaternion
import random


class GCN_Discriminator(nn.Module):
    """
    ⁘ ∴ : ·
    C: data channel
    V: data vertex
    skeleton_kernel: 2* | 3
    time_kernel: 4* | 6
    symmetry_break: none | random | radial*
    spectral_norm: false* | true
    activation: leakyrelu* | relu | elu
    motion_flow: false | true*
    base_channel: base channel number. 64 | 32*
    skeleton_path: path to skeleton graph adjacency
    """

    def __init__(self,
                 C,
                 V,
                 skeleton_kernel=2,
                 time_kernel=4,
                 symmetry_break='radial',
                 spectral_norm=True,
                 activation='leakyrelu',
                 motion_flow=False,
                 base_channel=32,
                 noise_std=0,
                 skeleton_path=None,
                 use_graph_conv=True,
                 time_pooling=True,
                 num_class=60,
                 set_size=0):
        super().__init__()
        # check
        assert skeleton_path is not None
        assert skeleton_kernel in [1, 2, 3] and time_kernel in [3, 4, 5, 6]
        assert activation in ['leakyrelu', 'relu', 'elu']
        assert base_channel in [16, 32, 64]

        self.C = C
        self.V = V
        self.skeleton_kernel = skeleton_kernel
        self.time_kernel = time_kernel
        self.symmetry_break = symmetry_break
        self.spectral_norm = spectral_norm
        self.activation = activation
        self.motion_flow = motion_flow
        self.base_channel = base_channel
        self.skeleton_path = skeleton_path
        self.noise_std = noise_std
        C = C * (1 + motion_flow)
        self.set_size = set_size

        if time_pooling:
            time_stride = 2
            time_pad = (time_kernel - 2) // 2
        else:
            time_stride = 1
            time_pad = (time_kernel - 1) // 2

        with h5py.File(self.skeleton_path, 'r') as f:
            self.skeleton = f['adjacency'][:]
            self.parent = f['parent'][:] if 'parent' in f.keys() else None

        # get kernels
        g = self.skeleton
        self.A44, _ = get_graph_kernel(skeleton_kernel, 1, spectral_norm, g, symmetry_break)  # 2,25,25
        self.A43, g = get_graph_kernel(skeleton_kernel, 2, spectral_norm, g, symmetry_break)  # 2,11,25
        self.A33, _ = get_graph_kernel(skeleton_kernel, 1, spectral_norm, g, symmetry_break)  # 2,11,11
        self.A32, g = get_graph_kernel(skeleton_kernel, 2, spectral_norm, g, symmetry_break)  # 2,5,11
        self.A22, _ = get_graph_kernel(skeleton_kernel, 1, spectral_norm, g, symmetry_break)  # 2,5,5
        self.A21, g = get_graph_kernel(skeleton_kernel, 2, spectral_norm, g, symmetry_break)  # 2,1,5
        self.A11, _ = get_graph_kernel(1, 1, spectral_norm, g, symmetry_break)  # 1,1,1
        self.A00 = np.array([[[1.0]]])

        def get_conv(D_in, D_out, adj, kernel, stride, padding, bias=False, pooling=False):
            return GraphConv(
                extra_dim=1,
                in_channels=D_in,
                out_channels=D_out,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=bias,
                spectral_norm=self.spectral_norm,
                mode='manual',
                computation_kernel=adj,
                symmetry_break=self.symmetry_break,
                normalize='group',
                pooling=pooling,
                return_graph=False,
                use_graph_conv=use_graph_conv)

        def get_act():
            if self.activation == 'relu':
                return nn.ReLU()
            elif self.activation == 'leakyrelu':
                return nn.LeakyReLU(0.2)
            elif self.activation == 'elu':
                return nn.ELU()
            else:
                raise ValueError(f'activation {self.activation} not supported')

        def get_down(D_in, D_out, adj):
            return nn.Sequential(
                get_conv(
                    D_in,
                    D_out,
                    adj, (skeleton_kernel, time_kernel),
                    stride=(2, time_stride),
                    padding=(0, time_pad),
                    bias=False), get_act())

        c1, c2, c3, c4, c5 = [int(base_channel * c) for c in [1, 2, 4, 8, 16]]

        # 3   x ⁘ x 64 => 32  x ⁘ x 64
        self.conv0 = get_conv(C, c1, self.A44, (skeleton_kernel, 5),
                                  stride=(1, 1), padding=(0, 2), bias=True)
        self.relu0 = get_act()

        self.down1 = get_down(c1, c2, self.A43) # 32  x ⁘ x 64 => 64  x ∴ x 32
        self.down2 = get_down(c2, c3, self.A32) # 64  x ∴ x 32 => 128 x : x 16
        self.down3 = get_down(c3, c4, self.A22)  # 128 x : x 16 => 256 x : x 8

        # 256 x : x 8  => 512 x 1 x 4
        self.conv4 = nn.Conv2d(c4, c5, (self.A22.shape[-1], time_kernel), stride=(2, time_stride), padding=(0, time_pad), bias=True)
        self.conv4 = nn.utils.spectral_norm(self.conv4)
        self.relu4 = get_act()

        # 512 x 1 x 4  => 1  x 1 x 1
        self.conv5 = nn.utils.spectral_norm(nn.Linear(c5, 1, bias=True))

        # class embedding
        self.embedding = nn.utils.spectral_norm(nn.Embedding(num_class, c5))
        nn.init.orthogonal_(self.embedding.weight)


    def forward(self, x, y):
        if self.noise_std:
            x = x + torch.zeros(*(x.size())).float().cuda().normal_(0, self.noise_std)

        # N, C, V, T
        q = torch.tensor(()).to(next(self.parameters()).device)
        if self.motion_flow:
            n, c, v, _ = x.size()
            m = torch.cat((torch.cuda.FloatTensor(n, c, v, 1).zero_(),
                           x[:, :, :, 1:] - x[:, :, :, :-1]), 3)
        else:
            m = torch.tensor(()).to(next(self.parameters()).device)

        x = torch.cat((x, q, m), dim=1)
        assert x.size(1) == self.C * (1 + self.motion_flow)

        x = self.relu0(self.conv0(x))
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = torch.sum(x, [2, 3])
        out = self.conv5(x)

        # conditional projection
        if y is not None:
            embed = self.embedding(y)
        x = out + torch.sum(embed * x, 1, keepdim=True)

        return x.view(x.size(0))


def get_graph_kernel(kernel, stride, spectral_norm, in_graph, symmetry_break, normalize='group'):
    c = GraphConv(
        extra_dim=1,
        in_channels=1,
        out_channels=1,
        kernel_size=kernel,
        stride=stride,
        spectral_norm=spectral_norm,
        mode='auto',
        in_graph=in_graph,
        symmetry_break=symmetry_break,
        normalize=normalize,
        return_graph=True)
    return c.computation_kernel, c.out_graph
