#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import torch
import torch.nn as nn
import pytest
from collections import OrderedDict

from speech_recognition.summaries import model_summary

class TestModelConv1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(1,1,1),
            nn.Conv1d(1,1,3,padding=1),
            nn.Conv1d(1,1,3,padding=1,stride=2),
        )

    def forward(self, x):
        return self.layers(x)


def test_conv1d_metrics():
    model = TestModelConv1D()
    dummy_input = torch.zeros((1,1,256))

    performance_result = model_summary(model, dummy_input, 'performance')

    print(performance_result)

    expected = OrderedDict([('Total MACs', 1408), ('Total Weights', 7), ('Total Activations', 896), ('Estimated Activations', 512)])

    assert performance_result == expected


if __name__  == "__main__":
    test_conv1d_metrics()
