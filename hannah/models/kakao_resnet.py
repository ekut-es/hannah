#
# Copyright (c) 2024 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
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
from typing import Sequence

import torch.nn as nn


# Model (from KakaoBrain: https://github.com/wbaek/torchskeleton)
class Mul(nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
    )


def resnet8(
    input_shape: Sequence[int],
    labels: int = 10,
    input_stride: int = 1,
    input_kernel: int = 3,
    width_multiplier: float = 1.0,
    **kwargs
):
    num_class = labels
    model = nn.Sequential(
        conv_bn(
            input_shape[1],
            int(64 * width_multiplier),
            kernel_size=input_kernel,
            stride=input_stride,
            padding=1,
        ),
        conv_bn(
            int(64 * width_multiplier),
            int(128 * width_multiplier),
            kernel_size=5,
            stride=2,
            padding=2,
        ),
        Residual(
            nn.Sequential(
                conv_bn(int(128 * width_multiplier), int(128 * width_multiplier)),
                conv_bn(int(128 * width_multiplier), int(128 * width_multiplier)),
            )
        ),
        conv_bn(
            int(128 * width_multiplier),
            int(256 * width_multiplier),
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.MaxPool2d(2),
        Residual(
            nn.Sequential(
                conv_bn(int(256 * width_multiplier), int(256 * width_multiplier)),
                conv_bn(int(256 * width_multiplier), int(256 * width_multiplier)),
            )
        ),
        conv_bn(
            int(256 * width_multiplier),
            int(128 * width_multiplier),
            kernel_size=3,
            stride=1,
            padding=0,
        ),
        nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        nn.Linear(int(128 * width_multiplier), num_class, bias=False),
        Mul(0.2),
    )

    return model
