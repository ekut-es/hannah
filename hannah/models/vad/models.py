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
import torch.nn as nn
import torch.nn.functional as F


class BottleneckVad(nn.Module):
    def __init__(
        self,
        conv1_features,
        conv1_size,
        conv2_features,
        conv2_size,
        conv3_features,
        conv3_size,
        fc_size,
        stride,
        batch_norm,
    ):
        super().__init__()
        self.batch_norm = batch_norm
        self.norm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, conv1_features, conv1_size, stride)
        self.norm2 = nn.BatchNorm2d(conv1_features)
        self.conv2a = nn.Conv2d(conv1_features, 4, 1)  # Bottleneck layer
        self.conv2b = nn.Conv2d(4, 4, conv2_size)  # Bottleneck layer
        self.conv2c = nn.Conv2d(4, conv2_features, 1)  # Bottleneck layer
        self.norm3 = nn.BatchNorm2d(conv2_features)
        self.conv3 = nn.Conv2d(conv2_features, conv3_features, conv3_size, stride)
        self.fc1 = nn.Linear(fc_size, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.batch_norm:
            x = self.norm1(x)
        x = F.leaky_relu(self.conv1(x))
        if self.batch_norm:
            x = self.norm2(x)
        x = F.leaky_relu(self.conv2a(x))
        x = F.leaky_relu(self.conv2b(x))
        x = F.leaky_relu(self.conv2c(x))
        if self.batch_norm:
            x = self.norm3(x)
        x = F.leaky_relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class SmallVad(nn.Module):
    def __init__(self, conv1_features, conv1_size, fc_size, stride, batch_norm):
        super().__init__()
        self.batch_norm = batch_norm
        self.norm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, conv1_features, conv1_size, stride=stride)
        self.fc1 = nn.Linear(fc_size, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.batch_norm:
            x = self.norm1(x)
        x = F.leaky_relu(self.conv1(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class SimpleVad(nn.Module):
    def __init__(
        self,
        conv1_features,
        conv1_size,
        conv2_features,
        conv2_size,
        conv3_features,
        conv3_size,
        fc_size,
        stride,
        batch_norm,
    ):
        super().__init__()
        self.batch_norm = batch_norm
        self.norm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, conv1_features, conv1_size, stride=stride)
        self.norm2 = nn.BatchNorm2d(conv1_features)
        self.conv2 = nn.Conv2d(
            conv1_features, conv2_features, conv2_size, stride=stride
        )
        self.norm3 = nn.BatchNorm2d(conv2_features)
        self.conv3 = nn.Conv2d(
            conv2_features, conv3_features, conv3_size, stride=stride
        )
        self.fc1 = nn.Linear(fc_size, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.batch_norm:
            x = self.norm1(x)
        x = F.leaky_relu(self.conv1(x))
        if self.batch_norm:
            x = self.norm2(x)
        x = F.leaky_relu(self.conv2(x))
        if self.batch_norm:
            x = self.norm3(x)
        x = F.leaky_relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class BottleneckVadModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        conv1_features = config["conv1_features"]
        conv1_size = config["conv1_size"]
        conv2_features = config["conv2_features"]
        conv2_size = config["conv2_size"]
        conv3_features = config["conv3_features"]
        conv3_size = config["conv3_size"]
        fc_size = config["fc_size"]
        stride = config["stride"]
        batch_norm = config["batch_norm"]

        self.net = BottleneckVad(
            conv1_features,
            conv1_size,
            conv2_features,
            conv2_size,
            conv3_features,
            conv3_size,
            fc_size,
            stride,
            batch_norm,
        )

    def forward(self, x):
        x = self.net.forward(x)
        return x


class SimpleVadModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        conv1_features = config["conv1_features"]
        conv1_size = config["conv1_size"]
        conv2_features = config["conv2_features"]
        conv2_size = config["conv2_size"]
        conv3_features = config["conv3_features"]
        conv3_size = config["conv3_size"]
        fc_size = config["fc_size"]
        stride = config["stride"]
        batch_norm = config["batch_norm"]

        self.net = SimpleVad(
            conv1_features,
            conv1_size,
            conv2_features,
            conv2_size,
            conv3_features,
            conv3_size,
            fc_size,
            stride,
            batch_norm,
        )

    def forward(self, x):
        x = self.net.forward(x)
        return x


class SmallVadModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        conv1_features = config["conv1_features"]
        conv1_size = config["conv1_size"]
        fc_size = config["fc_size"]
        stride = config["stride"]
        batch_norm = config["batch_norm"]

        self.net = SmallVad(conv1_features, conv1_size, fc_size, stride, batch_norm)

    def forward(self, x):
        x = self.net.forward(x)
        return x
