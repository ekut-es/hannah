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
import warnings
from collections import namedtuple
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import pytest
import torch

from hannah.backends import GRPCBackend
from hannah.backends.base import (
    ClassifierModule,
    InferenceBackendBase,
    ProfilingResult,
)
from hannah.models.ai8x.models_simplified import ai8x_search_space
from hannah.models.embedded_vision_net.models import embedded_vision_net, search_space
from hannah.nas.constraints.random_walk import RandomWalkConstraintSolver
from hannah.nas.export import to_onnx
from hannah.nas.functional_operators.op import ChoiceOp, Tensor, scope
from hannah.nas.functional_operators.operators import Conv2d, MaxPooling, Relu
from hannah.nas.parameters import IntScalarParameter
from hannah.nas.parameters.parametrize import set_parametrization
from hannah.nas.search.sampler.random_sampler import RandomSampler


class MockModule(torch.nn.Module): ...


def get_random_tensor_from_onnx(onnxmodel: onnx.ModelProto):
    # Check if model has any inputs
    if not onnxmodel.graph.input:
        raise ValueError("ONNX model has no inputs")

    # Get first input's shape, handling potential None/dynamic dimensions
    shape = []
    for dim in onnxmodel.graph.input[0].type.tensor_type.shape.dim:
        # Use dim_value if available, else default to 1 (or whatever default you want)
        shape.append(dim.dim_value if dim.dim_value else 1)

    return torch.randn(shape)


class SimpleTestModule(ClassifierModule):
    """Simple test module for the backends"""

    def __init__(self, model=None, onnxmodel=None):
        if model is None:
            magicmodel = MagicMock(spec=MockModule)
            magicmodel.onnx = onnxmodel
            super().__init__(None, model=magicmodel)
        else:
            super().__init__(None, model=model)

    def forward(self, x):
        return self.model(x)

    def get_example_input_array(self):
        return self.example_input_array

    def setup(self, stage):
        self.example_input_array = get_random_tensor_from_onnx(self.model.onnx)

    def prepare_data(self):
        pass

    def get_class_names(self):
        return ["test"]


def get_graph(seed):
    input = Tensor("input", (1, 3, 32, 32), axis=["N", "C", "H", "W"], grad=False)
    graph = ai8x_search_space(
        "test", input, num_classes=10, max_blocks=1, rng=np.random.default_rng(seed)
    )
    print(graph)

    torch.random.manual_seed(seed)

    warnings.warn("remove this when seedable randomsampling works")

    print("Init sampler")
    sampler = RandomSampler(None, graph.parametrization(flatten=True))

    print("Init solver")
    solver = RandomWalkConstraintSolver()

    print("Sampling:", end="")
    # for _ in range(10):
    for i in range(1):
        print(".", end="")

        sampled_params, keys = sampler.next_parameters()

        solver.solve(graph, sampled_params)

        parameters = solver.get_constrained_params(sampled_params)

        set_parametrization(parameters, graph.parametrization(flatten=True))
    return graph


def get_onnxmodel(seed):
    graph = get_graph(seed)
    onnx_model = to_onnx(graph)
    return onnx_model


def our_to_onnx(model: "type", filename: str = "") -> onnx.ModelProto:
    if hasattr(model, "onnx"):
        return model.onnx
    else:
        return to_onnx(model, filename)


@pytest.mark.xfail(reason="need to fix onnx model generation")
@pytest.mark.parametrize(
    "modelname, test_module",
    [
        # ("VirtualDevice", SimpleTestModule(model=get_graph())), # does not work for some reason
        ("VirtualDevice", SimpleTestModule(onnxmodel=get_onnxmodel(56))),
        # ("EVKit_V1", SimpleTestModule(onnxmodel=get_onnxmodel(56))), # does not work without a device
    ],
)
def test_virtual_device_profiling(modelname, test_module):
    test_module.prepare_data()
    test_module.setup("fit")

    x = test_module.get_example_input_array()

    backend = GRPCBackend(device_filter=[{"model": modelname}])
    with patch("hannah.backends.grpc.to_onnx", our_to_onnx):
        backend.prepare(test_module)
    result = backend.profile(x)

    ref = test_module(x)

    assert len(result.outputs) > 0, "did not return results"

    assert result.metrics["us_per_all"] is not None


@pytest.mark.parametrize(
    "seed",
    [],  # range(57) # dont need to find a seed everytime
)
def test_search_seed(seed):
    modelname, test_module = (
        "VirtualDevice",
        SimpleTestModule(onnxmodel=get_onnxmodel(seed)),
    )
    test_module.prepare_data()
    test_module.setup("fit")

    x = test_module.get_example_input_array()

    backend = GRPCBackend(device_filter=[{"model": modelname}])
    with patch("hannah.backends.grpc.to_onnx", our_to_onnx):
        backend.prepare(test_module)
    result = backend.profile(x)
    assert result.profile != dict(), "seed does not work"
