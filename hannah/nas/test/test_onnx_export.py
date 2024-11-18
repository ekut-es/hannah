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
import copy
import time
from pathlib import Path

import onnx

from hannah.models.embedded_vision_net.models import embedded_vision_net, search_space
from hannah.models.ai8x.models import ai8x_search_space
from hannah.nas.constraints.random_walk import RandomWalkConstraintSolver
from hannah.nas.export import to_onnx
from hannah.nas.functional_operators.op import ChoiceOp, Tensor, scope
from hannah.nas.functional_operators.operators import Conv2d, MaxPooling, Relu
from hannah.nas.parameters import IntScalarParameter
from hannah.nas.parameters.parametrize import set_parametrization
from hannah.nas.search.sampler.random_sampler import RandomSampler


@search_space
def conv3x3_relu(input):
    out_channels = 48
    weight = Tensor(
        "w1",
        (out_channels, input.shape()[1], 3, 3),
        axis=["O", "I", "kH", "kW"],
        grad=True,
    )
    conv = Conv2d()(input, weight)
    relu = Relu()(conv)
    return relu


@search_space
def op_choice(input):
    conv = Conv2d()(
        input,
        Tensor(
            "w1",
            (input.shape()[0], input.shape()[1], 3, 3),
            axis=["O", "I", "kH", "kW"],
            grad=True,
        ),
    )
    max_pool = MaxPooling(kernel_size=3, stride=1, padding=1)(input)

    out = ChoiceOp(conv, max_pool, switch=None)

    return out


def test_deepcopy():
    input = Tensor("input", (1, 3, 32, 32), axis=["N", "C", "H", "W"], grad=False)
    graph = conv3x3_relu(input)

    new_graph = copy.deepcopy(graph)

    print(graph)
    print(new_graph)


def test_export_conv2d():
    input = Tensor("input", (1, 3, 32, 32), axis=["N", "C", "H", "W"], grad=False)
    graph = conv3x3_relu(input)
    print(graph)

    onnx_model = to_onnx(graph)

    print(onnx.printer.to_text(onnx_model))


def test_export_choice():
    inp = Tensor("input", (1, 3, 32, 32), axis=["N", "C", "H", "W"], grad=False)
    graph = op_choice(inp)

    for choice in [0, 1]:
        graph.switch.current_value = choice

        onnx_model = to_onnx(graph)
        print(onnx.printer.to_text(onnx_model))


def test_export_embedded_vision_net():
    print("Start time: ", time.time())

    print("Init search space")
    input = Tensor("input", (1, 3, 32, 32), axis=["N", "C", "H", "W"], grad=False)

    graph = embedded_vision_net("test", input, num_classes=10, max_blocks=2)
    print(graph)

    print("Init sampler")
    sampler = RandomSampler(None, graph.parametrization(flatten=True))

    print("Init solver")
    solver = RandomWalkConstraintSolver()

    print("Sampling:", end="")
    for _ in range(10):
        print(".", end="")

        sampled_params, keys = sampler.next_parameters()

        solver.solve(graph, sampled_params)

        parameters = solver.get_constrained_params(sampled_params)

        set_parametrization(parameters, graph.parametrization(flatten=True))

        onnx_model = to_onnx(graph)

        print(onnx.printer.to_text(onnx_model))

    print("")
    print("Done")


def test_export_ai8x_net():
    print("Start time: ", time.time())

    print("Init search space")
    input = Tensor("input", (1, 3, 32, 32), axis=["N", "C", "H", "W"], grad=False)

    graph = ai8x_search_space("test", input, num_classes=10, max_blocks=1)
    print(graph)

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

        onnx_model = to_onnx(graph)

        this_dir = Path(
            __file__
        ).parent.parent.parent.parent  # hannah/nas/test/../../../../ => /
        path = this_dir / f"ai8x_net_{i}_20240907.onnx"

        print(onnx.printer.to_text(onnx_model))

        print("saving to location:", path)
        onnx.save(onnx_model, path)

    print("")
    print("Done")


if __name__ == "__main__":
    # test_deepcopy()
    # test_export_choice()
    # test_export_conv2d()
    # test_export_embedded_vision_net()
    test_export_ai8x_net()
