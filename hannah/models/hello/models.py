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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DSConv2d(nn.Module):
    def __init__(self, n_maps_in, n_maps_out, shape, strides=1, dropout_prob=0.5):
        super().__init__()

        # Depthwise separable convolutions need multiples of input as output
        assert n_maps_out % n_maps_in == 0

        pads = tuple(x // 2 for x in shape)

        self.conv1 = nn.Conv2d(
            n_maps_in, n_maps_in, shape, strides, groups=n_maps_in, padding=pads
        )
        self.batch_norm1 = nn.BatchNorm2d(n_maps_in)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.conv2 = nn.Conv2d(n_maps_in, n_maps_out, 1, 1)
        self.batch_norm2 = nn.BatchNorm2d(n_maps_out)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        return x


class DSCNNSpeechModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        width = config["width"]
        height = config["height"]
        n_labels = config["n_labels"]
        n_maps = config["n_feature_maps"]

        self.dropout_prob = config["dropout_prob"]

        x = Variable(torch.zeros(1, 1, height, width))

        self.convs = nn.ModuleList()
        self.ds_convs = nn.ModuleList()

        current_shape = x.shape

        count = 1
        while "conv{}_size".format(count) in config:
            conv_size = config["conv{}_size".format(count)]
            conv_stride = (1,) * len(conv_size)

            if "conv{}_stride".format(count) in config:
                conv_stride = config["conv{}_stride".format(count)]

            pads = tuple(x // 2 for x in conv_size)

            conv_dilation = (1,) * len(conv_size)

            conv = nn.Conv2d(
                current_shape[1],
                n_maps,
                conv_size,
                stride=conv_stride,
                dilation=conv_dilation,
                padding=pads,
            )

            self.convs.append(conv)
            x = conv(x)
            current_shape = x.shape

            batch_norm = nn.BatchNorm2d(n_maps)
            self.convs.append(batch_norm)

            dropout = nn.Dropout(self.dropout_prob)
            self.convs.append(dropout)

            count += 1

        count = 1
        while "ds_conv{}_size".format(count) in config:
            conv_size = config["ds_conv{}_size".format(count)]
            conv_stride = config["ds_conv{}_stride".format(count)]

            conv = DSConv2d(
                n_maps, n_maps, conv_size, conv_stride, config["dropout_prob"]
            )

            self.ds_convs.append(conv)
            x = conv(x)

            count += 1

        self.avg_pool = nn.AvgPool2d((x.size(2), x.size(3)), stride=0)
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)

        self.output = nn.Linear(x.shape[1], n_labels)
        x = self.output(x)

    def forward(self, x):
        layer = 0
        x = x.unsqueeze(1)

        for conv in self.convs:
            layer += 1
            x = conv(x)

        for conv in self.ds_convs:
            layer += 1
            x = conv(x)

        layer += 1
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)

        return x


class DNNSpeechModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        width = config["width"]
        height = config["height"]

        x = Variable(torch.zeros(1, 1, width, height))

        self.dense = nn.ModuleList()

        x = x.view(1, -1)
        last_size = x.size(1)
        count = 1
        while "dnn{}_size".format(count) in config:
            dnn_size = config["dnn{}_size".format(count)]
            dense = nn.Linear(last_size, dnn_size)
            self.dense.append(dense)
            x = dense(x)

            relu = nn.ReLU()
            self.dense.append(relu)
            x = relu(x)

            dropout = nn.Dropout(config["dropout_prob"])
            self.dense.append(dropout)
            x = dropout(x)

            last_size = x.view(1, -1).size(1)

            count += 1

        self.output = nn.Linear(last_size, n_labels)
        x = self.output(x)
        last_size = x.view(1, -1).size(1)

        sum = 0
        for param in self.parameters():
            sum += param.view(-1).size(0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.dense:
            x = layer(x)

        x = self.output(x)

        return x
