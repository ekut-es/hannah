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

"""Test the prepare and run methods for each of the hannah backends"""

import inspect

import pytest
import torch
import torch.nn as nn

import hannah.backends
from hannah.modules.base import ClassifierModule


def backends():
    """Iterates over the backends"""
    for item in inspect.getmembers(hannah.backends):
        if inspect.isclass(item[1]) and item[0] != "InferenceBackendBase":
            if item[1].available() and item[0] != "GRPCBackend":
                yield item[1]
            else:
                print(f"Skipping {item[0]} because it is not available")


class SimpleModule(ClassifierModule):
    """Simple test module for the backends"""

    def __init__(self):
        super().__init__(None, nn.Linear(1, 1), None, None)

    def forward(self, x):
        return self.model(x)

    def setup(self, stage):
        self.example_input_array = torch.tensor([1.0]).unsqueeze(0)

    def prepare_data(self):
        pass

    def get_class_names(self):
        return ["test"]


@pytest.mark.parametrize("backend", backends())
def test_backend(backend):
    test_module = SimpleModule()
    test_module.prepare_data()
    test_module.setup("fit")

    x = torch.tensor([1.0]).unsqueeze(0)

    backend = backend()
    backend.prepare(test_module)
    results = backend.run(x)

    ref = test_module(x)

    assert torch.allclose(results[0], ref)


@pytest.mark.parametrize("backend", backends())
def test_profile(backend):
    test_module = SimpleModule()
    test_module.prepare_data()
    test_module.setup("fit")

    x = torch.tensor([1.0]).unsqueeze(0)

    backend = backend()
    backend.prepare(test_module)
    result = backend.profile(x)

    ref = test_module(x)

    assert torch.allclose(result.outputs[0], ref)

    assert result.metrics["duration"] is not None
