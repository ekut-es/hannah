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
import pytest
from hannah.nas.functional_operators.executor import BasicExecutor


from hannah.nas.functional_operators.lazy import lazy
from hannah.nas.functional_operators.operators import Conv2d, Linear, Relu
from hannah.nas.functional_operators.op import Tensor, search_space
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter
from hannah.modules.vision.image_classifier import ImageClassifierModule
from hannah.nas.performance_prediction.nn_meter.predictor import NNMeterPredictor
from hannah.nas.constraints.random_walk import RandomWalkConstraintSolver
from hannah.nas.search.sampler.random_sampler import RandomSampler
from hannah.nas.parameters.parametrize import set_parametrization


from torch.optim import SGD
import torch.nn as nn
import torch


def conv2d(input, out_channels, kernel_size=1, stride=1, dilation=1):
    in_channels = input.shape()[1]
    weight = Tensor(
        name="weight",
        shape=(out_channels, in_channels, kernel_size, kernel_size),
        axis=("O", "I", "kH", "kW"),
        grad=True,
    )

    conv = Conv2d(stride, dilation)(input, weight)
    return conv


def relu(input):
    return Relu()(input)


def conv_relu(input, out_channels, kernel_size, stride):
    out = conv2d(
        input, out_channels=out_channels, stride=stride, kernel_size=kernel_size
    )
    out = relu(out)
    return out


def linear(input, out_features):
    input_shape = input.shape()
    in_features = input_shape[1] * input_shape[2] * input_shape[3]
    weight = Tensor(
        name="weight",
        shape=(in_features, out_features),
        axis=("in_features", "out_features"),
        grad=True,
    )

    out = Linear()(input, weight)
    return out


@search_space
def network(input):
    out = conv_relu(
        input,
        out_channels=IntScalarParameter(32, 64, name="out_channels"),
        kernel_size=CategoricalParameter([3, 5], name="kernel_size"),
        stride=CategoricalParameter([2], name="stride"),
    )
    out = conv_relu(
        out,
        out_channels=IntScalarParameter(16, 32, name="out_channels"),
        kernel_size=CategoricalParameter([3, 5], name="kernel_size"),
        stride=CategoricalParameter([1], name="stride"),
    )
    # out = conv_relu(out,
    #                 out_channels=IntScalarParameter(4, 64, name='out_channels'),
    #                 kernel_size=CategoricalParameter([1, 3, 5, 7], name="kernel_size"),
    #                 stride=CategoricalParameter([1, 2], name='stride'))

    out = linear(out, 10)

    return out


@pytest.mark.parametrize(
    "hardware_name",
    [
        "cortexA76cpu_tflite21",
        "adreno640gpu_tflite21",
        "adreno630_gpu",
        "myriadvpu_openvino2019r2",
    ],
)
def test_nn_meter(hardware_name):
    input = Tensor(name="input", shape=(1, 3, 32, 32), axis=("N", "C", "H", "W"))

    net = network(input)
    executor = BasicExecutor(net)
    executor.initialize()

    module = ImageClassifierModule(
        None,
        executor,
        None,
        None,
        None,
    )

    predictor = NNMeterPredictor(hardware_name)

    print("Init sampler")
    sampler = RandomSampler(None, net.parametrization(flatten=True))

    print("Init solver")
    solver = RandomWalkConstraintSolver()

    sampled_params, keys = sampler.next_parameters()

    solver.solve(net, sampled_params)

    parameters = solver.get_constrained_params(sampled_params)

    set_parametrization(parameters, net.parametrization(flatten=True))

    module.example_input_array = torch.randn(1, 3, 32, 32)

    results = predictor.predict(module)

    assert isinstance(results, dict)

    assert "duration" in results


if __name__ == "__main__":
    test_nn_meter("cortexA76cpu_tflite21")
