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
from z3 import And

from hannah.nas.constraints.constraint_model import ConstraintModel
from hannah.nas.dataflow.dataflow_graph import flatten
from hannah.nas.ops import batched_image_tensor
from hannah.nas.parameters.parameters import IntScalarParameter
from hannah.nas.test.network import residual_block


def test_constraint_model():
    # Create a network and flatten the graph
    input = batched_image_tensor(name="input")
    graph = residual_block(
        input,
        stride=IntScalarParameter(1, 2),
        output_channel=IntScalarParameter(4, 512, 4),
    )
    graph = flatten(graph)

    # build a constraint model
    cm = ConstraintModel()
    cm.build_model(graph)

    # retrieve constraint vars from cm for better clarity
    out_channel_main = cm.vars[
        "residual_block.0.block.0.conv_relu.2.Conv2d.0.weight.0.axis.o.size"
    ]
    out_channel_residual = cm.vars[
        "residual_block.0.residual.0.conv_relu.0.Conv2d.0.weight.0.axis.o.size"
    ]

    # Check assumptions for satisfiability
    assert cm.solver.check(out_channel_main <= 256).r > 0
    assert cm.solver.check(out_channel_main >= 1024).r < 0
    assert cm.solver.check(out_channel_main == 128).r > 0
    assert cm.solver.check(out_channel_main == 129).r < 0

    assert (
        cm.solver.check(And(out_channel_main >= 64), (out_channel_residual <= 128)).r
        > 0
    )
    assert (
        cm.solver.check(And(out_channel_main <= 64), (out_channel_residual >= 128)).r
        < 0
    )

    # one can find a possible (if current constraints SAT) configuration with .model()
    # cm.solver.check()
    # model = cm.solver.model()
    # print(model)

    # Iterative solving
    cm.solver.push()
    cm.solver.add(out_channel_main <= 256)

    assert cm.solver.check(out_channel_residual == 128).r > 0
    assert cm.solver.check(out_channel_residual == 512).r < 0

    # restore previous model
    cm.solver.pop()

    assert cm.solver.check(out_channel_residual == 128).r > 0
    assert cm.solver.check(out_channel_residual == 512).r > 0


def test_constraint_model_parameters():
    input = batched_image_tensor(name="input")
    graph = residual_block(
        input,
        stride=IntScalarParameter(1, 2),
        output_channel=IntScalarParameter(4, 512, 4),
    )
    graph = flatten(graph)

    # build a constraint model
    cm = ConstraintModel()
    cm.build_model(graph)

    params = graph.parametrization(flatten=True)

    for i in range(1):
        cm.solver.push()
        for name, param in params.items():
            assert name in cm.vars
            cm.solver.add(cm.vars[name] == param.sample().item())

        print("Sat: ", cm.solver.check())
        cm.solver.pop()


if __name__ == "__main__":
    test_constraint_model()
    test_constraint_model_parameters()
