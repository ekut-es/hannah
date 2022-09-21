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


class ConvNet(nn.Module):
    def __init__(self, convolutions, pooling, linear, dropout, qconfig=None):
        super().__init__()
        self.convolutions = convolutions
        self.pooling = pooling
        self.dropout = dropout
        self.flatten = nn.Flatten()
        self.linear = linear
        self.qconfig = qconfig
        if qconfig:
            self.activation_post_process = self.qconfig.activation()

    def forward(self, x):
        if hasattr(self, "activation_post_process"):
            x = self.activation_post_process(x)
        x = self.convolutions(x)
        x = self.pooling(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
