#
# Copyright (c) 2023 Hannah contributors.
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
import torch
from torch import nn

from ..utils import next_power_of2


class ApproximateGlobalAveragePooling1D(nn.Module):
    """A global average pooling layer, that divides by the next power of 2 instead of true number of elements"""

    def __init__(self, size, qconfig=None):
        super().__init__()

        self.size = size
        self.divisor = next_power_of2(size)
        self.qconfig = qconfig

        # FIXME: Pass correct scale factor for prequantize
        if qconfig:
            self.activation_post_process = qconfig.activation()
            self.prequantize = qconfig.accumulator()

    def forward(self, x):
        """

        Args:
          x:

        Returns:

        """
        if hasattr(self, "prequantize"):
            x = self.prequantize(x)

        x = torch.sum(x, dim=2, keepdim=True)
        x = x / self.divisor

        if hasattr(self, "activation_post_process"):
            x = self.activation_post_process(x)

        return x


class ApproximateGlobalAveragePooling2D(nn.Module):
    """A global average pooling layer, that divides by the next power of 2 instead of true number of elements"""

    def __init__(self, size, qconfig=None):
        super().__init__()

        self.size = size
        self.divisor = next_power_of2(size)
        self.qconfig = qconfig
        if qconfig:
            self.activation_post_process = qconfig.activation()

    def forward(self, x):
        """

        Args:
          x:

        Returns:

        """
        x = torch.sum(x, dim=[2, 3], keepdim=True)
        x = x / self.divisor

        if hasattr(self, "activation_post_process"):
            x = self.activation_post_process(x)

        return x
