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
import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import onnx
import spox
import spox.opset.ai.onnx.v20 as op
import torch
from optree import tree_map
from spox._attributes import AttrFloat32, AttrInt64
from spox._fields import BaseAttributes, BaseInputs, BaseOutputs, Var
from spox._function import to_function
from spox._node import OpType
from spox._standard import StandardNode

from hannah.utils import pair, quadruple, single, triple

from ..core.expression import Expression
from ..functional_operators.executor import BasicExecutor
from ..functional_operators.op import ChoiceOp, Op, Tensor
from ..functional_operators.operators import (
    AdaptiveAvgPooling,
    Add,
    AvgPooling,
    BatchNorm,
    Conv1d,
    Conv2d,
    Dropout,
    Identity,
    InterleaveChannels,
    Linear,
    MaxPooling,
    Relu,
    Requantize,
    SelfAttention2d,
)
from ..parameters.parameters import Parameter

logger = logging.getLogger(__name__)


class _BatchNormalization(StandardNode):
    @dataclass
    class Attributes(BaseAttributes):
        epsilon: AttrFloat32
        momentum: AttrFloat32
        training_mode: AttrInt64

    @dataclass
    class Inputs(BaseInputs):
        X: Var
        scale: Var
        B: Var
        input_mean: Var
        input_var: Var

    @dataclass
    class Outputs(BaseOutputs):
        Y: Var

    op_type = OpType("BatchNormalization", "", 15)

    attrs: Attributes
    inputs: Inputs
    outputs: Outputs


def batch_normalization(
    X: Var,
    scale: Var,
    B: Var,
    input_mean: Var,
    input_var: Var,
    *,
    epsilon: float = 9.999999747378752e-06,
    momentum: float = 0.8999999761581421,
    training_mode: int = 0,
) -> Var:
    assert (
        not training_mode
    ), "Only support export of batch normalization for inference."

    return _BatchNormalization(
        _BatchNormalization.Attributes(
            epsilon=AttrFloat32(epsilon, name="epsilon"),
            momentum=AttrFloat32(momentum, name="momentum"),
            training_mode=AttrInt64(training_mode, name="training_mode"),
        ),
        _BatchNormalization.Inputs(
            X=X,
            scale=scale,
            B=B,
            input_mean=input_mean,
            input_var=input_var,
        ),
    ).outputs.Y


def eval(exp_tree: Any) -> Any:
    "Recursively evaluate expressions organized as a pytree"

    def do_eval(exp):
        if isinstance(exp, Expression):
            return exp.evaluate()
        elif isinstance(exp, Parameter):
            return exp.evaluate()
        return exp

    return tree_map(do_eval, exp_tree)


def to_onnx(model: Union[BasicExecutor, Op], filename: str = "") -> onnx.ModelProto:
    if isinstance(model, BasicExecutor):
        model = model.output

    worklist = [(model, False)]

    visited = set()

    input_tensors = []
    intializer_tensors = []

    node_cache = {}
    while worklist:
        node, visited_operands = worklist.pop()
        if node in visited:
            continue
        if visited_operands:
            visited.add(node)
            if isinstance(node, Tensor):
                res = spox.argument(
                    spox.Tensor(eval(node.dtype).as_numpy(), eval(node.shape())),
                )
                input_tensors.append(node)
                if node.grad:
                    # create an initializer instead of input
                    # call hannah.nas
                    intializer_tensors.append(node)
            elif isinstance(node, Relu):
                res = op.relu(node_cache[node.operands[0]])
            elif isinstance(node, Conv1d):
                res = op.conv(
                    node_cache[node.operands[0]],
                    node_cache[node.operands[1]],
                    dilations=single(eval(node.dilation)),
                    group=eval(node.groups),
                    pads=triple(eval(node.padding)),
                    strides=single(eval(node.stride)),
                    kernel_shape=pair(
                        eval(node.kernel_size)
                    ),  # its weird that it is a pair?
                )
            elif isinstance(node, Conv2d):
                res = op.conv(
                    node_cache[node.operands[0]],
                    node_cache[node.operands[1]],
                    dilations=pair(eval(node.dilation)),
                    group=eval(node.groups),
                    pads=quadruple(eval(node.padding)),
                    strides=pair(eval(node.stride)),
                    kernel_shape=pair(eval(node.kernel_size)),
                )
            elif isinstance(node, Linear):
                input_shape = eval(node.operands[0].shape_fun())
                inp_node = node.operands[0]
                inp = node_cache[inp_node]

                if len(input_shape) > 2:
                    inp = op.flatten(inp, axis=1)

                res = op.matmul(inp, node_cache[node.operands[1]])
            elif isinstance(node, MaxPooling):
                res, _ = op.max_pool(
                    node_cache[node.operands[0]],
                    kernel_shape=pair(eval(node.kernel_size)),
                    pads=quadruple(eval(node.padding)),
                    strides=pair(eval(node.stride)),
                )
            elif isinstance(node, AvgPooling):
                res = op.average_pool(
                    node_cache[node.operands[0]],
                    kernel_shape=pair(eval(node.kernel_size)),
                    pads=quadruple(eval(node.padding)),
                    strides=pair(eval(node.stride)),
                )
            elif isinstance(node, AdaptiveAvgPooling):
                output_size = eval(node.shape())
                if all((x == 1 for x in output_size[2:])):
                    res = op.global_average_pool(node_cache[node.operands[0]])
                else:
                    raise Exception(
                        "AdaptiveAvgPooling is not supported for output sizes other than 1x1"
                    )

            elif isinstance(node, Dropout):
                # Only export for inference
                res = node_cache[node.operands[0]]

            elif isinstance(node, Add):
                operands = [node_cache[operand] for operand in node.operands]
                res = op.add(node_cache[node.operands[0]], node_cache[node.operands[1]])
                operands = operands[2:]
                while len(operands) > 1:
                    res = op.add(res, operands.pop())

            elif isinstance(node, BatchNorm):
                channels = eval(node.shape()[1])
                dtype = "float32"
                scale = op.constant(value=np.ones(channels, dtype=dtype))
                bias = op.constant(value=np.zeros(channels, dtype=dtype))

                X = node_cache[node.operands[0]]
                mean = node_cache[node.operands[1]]
                var = node_cache[node.operands[2]]

                # print(mean, var)

                epsilon = 1e-5

                res = batch_normalization(
                    X, scale, bias, mean, var, epsilon=epsilon, training_mode=0
                )

            elif isinstance(node, InterleaveChannels):
                step_size = eval(node.step_size)
                x = node_cache[node.operands[0]]

                # FIXME: channel interleaving should be a function, but it is not supported yet
                # @to_function("InterleaveChannels", "hannah")
                # def interleave_channels2d(x: spox.Var, step_size: spox.Var):

                #     shape = op.shape(x)

                #     n, c, h, w = op.split(shape, axis=0, num_outputs=4)

                #     reshape1 = op.reshape(n, (n, -1, step_size, h, w))

                #     # Transpose the tensor
                #     transpose1 = op.transpose(reshape1, (0, 2, 1, 3, 4))

                #     # Reshape the tensor back to the original shape
                #     reshape2 = op.reshape(transpose1, shape)

                #     return [reshape2]

                #     return [x]
                # res,  = interleave_channels2d(x, op.constant(value_int=step_size))

                shape = eval(node.operands[0].shape_fun())

                assert shape[1] % step_size == 0

                reshape1 = op.reshape(
                    x,
                    op.constant(
                        value_ints=(shape[0], shape[1] // step_size, step_size, -1)
                    ),
                )
                transpose = op.transpose(reshape1, perm=(0, 2, 1, 3))
                reshape2 = op.reshape(transpose, op.constant(value_ints=shape))

                res = reshape2

            elif isinstance(node, SelfAttention2d):
                raise NotImplementedError("SelfAttention2d is not supported yet")
            elif isinstance(node, Identity):
                res = node_cache[node.operands[0]]
            elif isinstance(node, ChoiceOp):
                switch = eval(node.switch)
                res = node_cache[node.options[switch]]
            elif isinstance(node, Requantize):
                dtype = eval(node.dtype)
                scale = eval(node.scale)
                zero_point = eval(node.zero_point)
                ch_axis = eval(node.ch_axis)
                if dtype == torch.quint8:
                    zero_point = zero_point.astype(np.uint8)
                elif dtype != torch.qint8:
                    zero_point = zero_point.astype(np.int8)
                else:
                    raise NotImplementedError(
                        f"Requantize is only supported for int8(types), got {dtype}"
                    )

                zero_point = op.constant(value=zero_point)
                scale = op.constant(value=scale)

                res = op.quantize_linear(
                    node_cache[node.operands[0]],
                    scale,
                    zero_point,
                    axis=ch_axis,
                )

                res = op.dequantize_linear(res, scale, zero_point, axis=ch_axis)

            else:
                raise NotImplementedError(f"Unsupported operator: {node}")

            node_cache[node] = res
        else:
            worklist.append((node, True))
            for operand in node.operands:
                worklist.append((operand, False))
            if isinstance(node, ChoiceOp):
                switch = eval(node.switch)
                worklist.append((node.options[switch], False))

    inputs = {}
    for idx, node in enumerate(input_tensors):
        node_id = f"{node.id}_{idx}"
        inputs[node_id] = node_cache[node]
    outputs = {}
    outputs[model.id] = node_cache[model]

    initializers = {}
    for idx, node in enumerate(intializer_tensors):
        node_id = f"{node.id}_{idx}"
        initializers[node_id] = node_cache[node]

    return spox.build(inputs=inputs, outputs=outputs)
