#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def conv_bn(inp, oup, stride):
    """

    Args:
      inp:
      oup:
      stride:

    Returns:

    """
    return nn.Sequential(
        nn.Conv1d(inp, oup, 160, stride, 4, bias=False),
        nn.BatchNorm1d(oup),
        nn.ReLU6(inplace=True),
        nn.MaxPool1d(8, 8),
    )


def conv_1x1_bn(inp, oup):
    """

    Args:
      inp:
      oup:

    Returns:

    """
    return nn.Sequential(
        nn.Conv1d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm1d(oup), nn.ReLU6()
    )


class InvertedResidual(nn.Module):
    """ """

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv1d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU6(),
                # pw-linear
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv1d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU6(),
                # dw
                nn.Conv1d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU6(),
                # pw-linear
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
            )

    def forward(self, x):
        """

        Args:
          x:

        Returns:

        """
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class RawSpeechModel(nn.Module):
    """Speech Recognition on RAW Data using Wolfgang Fuhls Networks"""

    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        width = config["input_length"]
        act = config["act"] if "act" in config else None

        last_size = 0

        x = Variable(torch.zeros(1, 1, width))

        self.convolutions = nn.ModuleList()
        self.dense = nn.ModuleList()

        count = 1
        while "conv{}_size".format(count) in config:

            fmap_in_name = "n_feature_maps_{}".format(count)
            fmap_out_name = "n_feature_maps_{}".format(count + 1)
            conv_size_name = "conv{}_size".format(count)
            conv_stride_name = "conv{}_stride".format(count)
            conv_dilation_name = "conv{}_dilation".format(count)

            pool_size_name = "pool{}_size".format(count)
            pool_stride_name = "pool{}_stride".format(count)

            if conv_size_name in config:
                n_feature_maps_in = config[fmap_in_name]
                n_feature_maps_out = config[fmap_out_name]
                conv_size = config[conv_size_name]
                conv_stride = config[conv_stride_name]
                conv_dilation = 1

                if conv_dilation_name in config:
                    conv_dilation = config[conv_dilation_name]

                pad = conv_size * conv_dilation // 2
                conv = nn.Conv1d(
                    n_feature_maps_in,
                    n_feature_maps_out,
                    conv_size,
                    conv_stride,
                    padding=pad,
                    dilation=conv_dilation,
                )
                x = conv(x)
                self.convolutions.append(conv)

                if act == "relu":
                    activation = nn.ReLU()
                    self.convolutions.append(activation)
                    x = activation(x)

                dropout = nn.Dropout(config["dropout_prob"])
                self.convolutions.append(dropout)
                x = dropout(x)

            last_size = x.view(1, -1).size(1)

            if pool_size_name in config:
                pool_size = config[pool_size_name]
                pool_stride = config[pool_stride_name]
                pool = nn.MaxPool1d(pool_size, pool_stride)
                x = pool(x)
                self.convolutions.append(pool)

            last_size = x.view(1, -1).size(1)

            count += 1

        x = x.view(1, -1)

        count = 1
        while "dnn{}_size".format(count) in config:
            dnn_size = config["dnn{}_size".format(count)]

            dnn = nn.Linear(last_size, dnn_size)
            self.dense.append(dnn)

            x = dnn(x)
            last_size = x.view(1, -1).size(1)

            count += 1

            if act == "relu":
                activation = nn.ReLU()
                self.dense.append(activation)
                x = activation(x)

            dropout = nn.Dropout(config["dropout_prob"])
            self.dense.append(dropout)
            x = dropout(x)

        self.output = nn.Linear(last_size, n_labels)
        x = self.output(x)
        last_size = x.view(1, -1).size(1)

    def forward(self, x):
        """

        Args:
          x:

        Returns:

        """
        num = 0
        for layer in self.convolutions:
            x = layer(x)
            num += 1

        x = x.view(x.size(0), -1)
        for layer in self.dense:
            x = layer(x)
            num += 1

        x = self.output(x)
        return x


class RawSpeechModelInvertedResidual(nn.Module):
    """ """

    def __init__(self, config):
        super().__init__()

        n_class = config["n_labels"]
        input_size = config["input_length"]
        width_mult = config["width_mult"]
        dropout_prob = config["dropout_prob"]

        block = InvertedResidual
        input_channel = config["input_channel"]
        last_channel = config["last_channel"]
        #        interverted_residual_setting = [
        #            # t, c, n, s
        #            [1, 16, 1, 1],
        #            [6, 24, 2, 2],
        #            [6, 32, 3, 2],
        #            [6, 64, 4, 2],
        #            [6, 64, 3, 1],
        #            [6, 64, 3, 2],
        #            [6, 64, 1, 1],
        #        ]

        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 2],
            [1, 24, 1, 2],
            [1, 32, 1, 2],
            [1, 64, 1, 2],
            [1, 64, 1, 2],
            [1, 64, 1, 2],
            [1, 64, 1, 2],
            [1, 64, 1, 2],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = (
            int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        )
        self.features = [conv_bn(1, input_channel, 2)]

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(input_channel, output_channel, s, expand_ratio=t)
                    )
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, expand_ratio=t)
                    )
                input_channel = output_channel

        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob), nn.Linear(self.last_channel, n_class)
        )

        self._initialize_weights()

    def forward(self, x):
        """

        Args:
          x:

        Returns:

        """
        x = self.features(x)
        x = x.mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """ """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
