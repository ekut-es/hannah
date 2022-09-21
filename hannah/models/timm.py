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
import timm
from torch import nn


class TimmModel(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        labels: int,
        name: str,
        pretrained: bool = False,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.model = timm.create_model(name, num_classes=labels, pretrained=pretrained)
        self.input_shape = input_shape

    def forward(self, x):
        return self.model(x)
