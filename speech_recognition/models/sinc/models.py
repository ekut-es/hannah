from typing import Dict, Any


import torch
import torch.nn as nn
from torch.nn.modules.utils import _single
from torch.autograd import Variable
import torch.nn.functional as F

import logging

msglogger = logging.getLogger()

import numpy as np
import matplotlib.pyplot as plt

from speech_recognition.features import SincConv
from ..utils import ConfigType, SerializableModule, next_power_of2


########################## Activation Function ################################


class Sinc_Act(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):

        return torch.log10(torch.abs(input) + 1)


############################### SincConv Block ################################


class SincConvBlock(nn.Module):
    def __init__(self, N_filt, filt_len, bn_len, avgpool_len, SR, stride):
        super(SincConvBlock, self).__init__()

        self.layer = nn.Sequential(
            SincConv(N_filt, filt_len, SR, stride=stride, padding=filt_len // 2),
            Sinc_Act(),
            nn.BatchNorm1d(bn_len),
            nn.AvgPool1d(avgpool_len),
        )

    def forward(self, x):

        out = self.layer(x)

        return out


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


class SincNet(SerializableModule):
    def __init__(self, config):
        super(SincNet, self).__init__()

        self.num_classes = config["num_classes"]

        self.cnn_N_filt = config["cnn_N_filt"]
        self.cnn_filt_len = config["cnn_filt_len"]
        self.cnn_bn_len = config["cnn_bn_len"]
        self.cnn_avgpool_len = config["cnn_avgpool_len"]
        self.SR = config["samplingrate"]
        self.cnn_stride = config["cnn_stride"]

        self.dsconv_N_filt = config["dsconv_N_filt"]
        self.dsconv_filt_len = config["dsconv_filt_len"]
        self.dsconv_stride = config["dsconv_stride"]
        self.dsconv_pcstride = config["dsconv_pcstride"]
        self.dsconv_groups = config["dsconv_groups"]
        self.dsconv_avg_pool_len = config["dsconv_avg_pool_len"]
        self.dsconv_bn_len = config["dsconv_bn_len"]
        self.dsconv_spatDrop = config["dsconv_spatDrop"]
        self.dsconv_num = len(config["dsconv_N_filt"])

        self.SincNet = SincConvBlock(
            self.cnn_N_filt,
            self.cnn_filt_len,
            self.cnn_bn_len,
            self.cnn_avgpool_len,
            self.SR,
            self.cnn_stride,
        )
        self.GDSBlocks = nn.ModuleList([])

        for i in range(self.dsconv_num):
            if i == 0:
                self.GDSBlocks.append(
                    GDSConvBlock(
                        self.cnn_N_filt,
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
        self.fc = nn.Linear(self.dsconv_N_filt[self.dsconv_num - 1], self.num_classes)

    def forward(self, x):

        batch = x.shape[0]
        x = x.view(x.shape[0], 1, x.shape[2])
        x = self.SincNet(x)

        for i in range(self.dsconv_num):
            x = self.GDSBlocks[i](x)

        x = self.Global_avg_pool(x)

        x = x.view(batch, -1)
        x = self.fc(x)

        return x


###############################################################################


configs = {
    ConfigType.SINC1.value: dict(
        features="raw",
        num_classes=12,
        cnn_N_filt=40,
        cnn_filt_len=101,
        cnn_bn_len=40,
        cnn_avgpool_len=2,
        cnn_stride=8,
        dsconv_N_filt=(162, 162, 162, 162, 162),
        dsconv_filt_len=(25, 9, 9, 9, 9),
        dsconv_stride=(2, 1, 1, 1, 1),
        dsconv_pcstride=(1, 1, 1, 1, 1),
        dsconv_groups=(1, 2, 3, 2, 3),
        dsconv_avg_pool_len=(2, 2, 2, 2, 2),
        dsconv_bn_len=(162, 162, 162, 162, 162),
        dsconv_spatDrop=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    ),
    ConfigType.SINC2.value: dict(
        features="raw",
        num_classes=12,
        cnn_N_filt=40,
        cnn_filt_len=101,
        cnn_bn_len=40,
        cnn_avgpool_len=2,
        cnn_stride=32,
        dsconv_N_filt=(160, 160, 160, 160, 160),
        dsconv_filt_len=(25, 9, 9, 9, 7),
        dsconv_stride=(2, 1, 1, 1, 1),
        dsconv_pcstride=(1, 1, 1, 1, 1),
        dsconv_groups=(1, 4, 8, 4, 8),
        dsconv_avg_pool_len=(2, 2, 1, 1, 1),
        dsconv_bn_len=(160, 160, 160, 160, 160),
        dsconv_spatDrop=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    ),
    ConfigType.SINC3.value: dict(
        features="raw",
        num_classes=12,
        cnn_N_filt=40,
        cnn_filt_len=101,
        cnn_bn_len=40,
        cnn_avgpool_len=2,
        cnn_stride=16,
        dsconv_N_filt=(160, 160, 160, 160, 160),
        dsconv_filt_len=(25, 9, 9, 9, 6),
        dsconv_stride=(2, 1, 1, 1, 1),
        dsconv_pcstride=(1, 1, 1, 1, 1),
        dsconv_groups=(1, 4, 4, 4, 4),
        dsconv_avg_pool_len=(2, 2, 2, 2, 2),
        dsconv_bn_len=(160, 160, 160, 160, 160),
        dsconv_spatDrop=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    ),
    ConfigType.SINC4.value: dict(
        features="raw",
        num_classes=12,
        cnn_N_filt=40,
        cnn_filt_len=101,
        cnn_bn_len=40,
        cnn_avgpool_len=2,
        cnn_stride=16,
        dsconv_N_filt=(160, 160, 160, 160, 160),
        dsconv_filt_len=(25, 9, 9, 9, 7),
        dsconv_stride=(2, 1, 1, 1, 1),
        dsconv_pcstride=(1, 1, 1, 1, 1),
        dsconv_groups=(1, 2, 4, 2, 4),
        dsconv_avg_pool_len=(2, 2, 2, 2, 2),
        dsconv_bn_len=(160, 160, 160, 160, 160),
        dsconv_spatDrop=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    ),
    ConfigType.SINC5.value: dict(
        features="raw",
        num_classes=12,
        cnn_N_filt=40,
        cnn_filt_len=101,
        cnn_bn_len=40,
        cnn_avgpool_len=2,
        cnn_stride=16,
        dsconv_N_filt=(160, 160, 160, 160, 160),
        dsconv_filt_len=(25, 9, 9, 9, 7),
        dsconv_stride=(2, 1, 1, 1, 1),
        dsconv_pcstride=(1, 1, 1, 1, 1),
        dsconv_groups=(1, 4, 8, 4, 8),
        dsconv_avg_pool_len=(2, 2, 2, 2, 2),
        dsconv_bn_len=(160, 160, 160, 160, 160),
        dsconv_spatDrop=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    ),
    ConfigType.SINC6.value: dict(
        features="raw",
        num_classes=12,
        cnn_N_filt=40,
        cnn_filt_len=101,
        cnn_bn_len=40,
        cnn_avgpool_len=2,
        SR=16000,
        cnn_stride=32,
        dsconv_N_filt=(160, 160, 160, 160, 160),
        dsconv_filt_len=(25, 9, 9, 8, 1),
        dsconv_stride=(2, 1, 1, 1, 1),
        dsconv_pcstride=(1, 1, 1, 1, 1),
        dsconv_groups=(1, 4, 8, 4, 8),
        dsconv_avg_pool_len=(2, 2, 2, 2, 2),
        dsconv_bn_len=(160, 160, 160, 160, 160),
        dsconv_spatDrop=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    ),
    ConfigType.SINC7.value: dict(
        features="raw",
        num_classes=12,
        cnn_N_filt=40,
        cnn_filt_len=101,
        cnn_bn_len=40,
        cnn_avgpool_len=2,
        cnn_stride=32,
        dsconv_N_filt=(160, 160, 160, 160, 160),
        dsconv_filt_len=(25, 9, 9, 8, 1),
        dsconv_stride=(2, 1, 1, 1, 1),
        dsconv_pcstride=(1, 1, 1, 1, 1),
        dsconv_groups=(1, 2, 4, 2, 4),
        dsconv_avg_pool_len=(2, 2, 1, 1, 1),
        dsconv_bn_len=(160, 160, 160, 160, 160),
        dsconv_spatDrop=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    ),
    ConfigType.SINC8.value: dict(
        features="raw",
        num_classes=12,
        cnn_N_filt=40,
        cnn_filt_len=101,
        cnn_bn_len=40,
        cnn_avgpool_len=2,
        cnn_stride=32,
        dsconv_N_filt=(160, 160, 160, 160, 160),
        dsconv_filt_len=(25, 9, 9, 9, 8),
        dsconv_stride=(2, 1, 1, 1, 1),
        dsconv_pcstride=(1, 1, 1, 1, 1),
        dsconv_groups=(1, 4, 4, 4, 4),
        dsconv_avg_pool_len=(2, 2, 1, 1, 1),
        dsconv_bn_len=(160, 160, 160, 160, 160),
        dsconv_spatDrop=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    ),
}
