import torch
import torch.nn as nn
from torch.nn.modules.utils import _single
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from ..utils import next_power_of2


############################# DS Conv Block ###################################
class GDSConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        pcstride,
        groups,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super(GDSConv, self).__init__()

        self.layer1 = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding=padding,
        )  # depthwise convolution with k*1 filters
        self.layer2 = nn.Conv1d(
            in_channels, out_channels, 1, pcstride, 0, 1, groups=groups, bias=bias
        )
        # pointwise convolutions with 1*c/g filters

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class GDSConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        pcstride,
        groups,
        avg_pool_len,
        bn_len,
        spatDrop,
    ):
        super(GDSConvBlock, self).__init__()

        self.layer = nn.Sequential(
            GDSConv(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                pcstride,
                groups,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(bn_len),
            nn.AvgPool1d(avg_pool_len, padding=avg_pool_len // 2),
            nn.Dropout(spatDrop),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


############################ Combined Final Block *****************************


class SincNet(nn.Module):
    def __init__(self, config):
        super(SincNet, self).__init__()
        self.height = config["height"]
        self.n_labels = config["n_labels"]
        self.dsconv_N_filt = config["dsconv_N_filt"]
        self.dsconv_filt_len = config["dsconv_filt_len"]
        self.dsconv_stride = config["dsconv_stride"]
        self.dsconv_pcstride = config["dsconv_pcstride"]
        self.dsconv_groups = config["dsconv_groups"]
        self.dsconv_avg_pool_len = config["dsconv_avg_pool_len"]
        self.dsconv_bn_len = config["dsconv_bn_len"]
        self.dsconv_spatDrop = config["dsconv_spatDrop"]
        self.dsconv_num = len(config["dsconv_N_filt"])

        self.GDSBlocks = nn.ModuleList([])

        for i in range(self.dsconv_num):
            if i == 0:
                self.GDSBlocks.append(
                    GDSConvBlock(
                        self.height,
                        self.dsconv_N_filt[i],
                        self.dsconv_filt_len[i],
                        self.dsconv_stride[i],
                        self.dsconv_pcstride[i],
                        self.dsconv_groups[i],
                        self.dsconv_avg_pool_len[i],
                        self.dsconv_bn_len[i],
                        self.dsconv_spatDrop[i],
                    )
                )

            else:
                self.GDSBlocks.append(
                    GDSConvBlock(
                        self.dsconv_N_filt[i - 1],
                        self.dsconv_N_filt[i],
                        self.dsconv_filt_len[i],
                        self.dsconv_stride[i],
                        self.dsconv_pcstride[i],
                        self.dsconv_groups[i],
                        self.dsconv_avg_pool_len[i],
                        self.dsconv_bn_len[i],
                        self.dsconv_spatDrop[i],
                    )
                )

        self.Global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.dsconv_N_filt[self.dsconv_num - 1], self.n_labels)

    def forward(self, x):

        batch = x.shape[0]
        for i in range(self.dsconv_num):
            x = self.GDSBlocks[i](x)

        x = self.Global_avg_pool(x)

        x = x.view(batch, -1)
        x = self.fc(x)

        return x
