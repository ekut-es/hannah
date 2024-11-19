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

import torch

from hannah.nas.functional_operators.data_type import IntType
from hannah.nas.functional_operators.executor import BasicExecutor
from hannah.nas.functional_operators.op import Tensor
from hannah.nas.functional_operators.quant import FixedQuantize
from hannah.nas.functional_operators.op import search_space


def test_fixed_quantize():
    for bits in range(3, 8):
        for signed in [True, False]:
            dtype = IntType(bits=bits, signed=signed)

            scale = -1.0 / 2 ** (bits - 1)
            zero_point = float(2 ** (bits - 1) - 1)

            quantizer = FixedQuantize(scale=scale, zero_point=zero_point, dtype=dtype)

            x = torch.tensor([0.0, 0.5, 0.75])

            x_quantized = quantizer.forward((x,))

            torch.testing.assert_close(
                x_quantized, torch.tensor([0.0, 0.5, 0.75]), rtol=1e-3, atol=1e-3
            )


def test_onnx_export():
    shape = (1, 1, 3, 3)
    input = Tensor(name="input", shape=shape, axis=("N", "C", "H", "W"), grad=False)

    @search_space
    def network(input):
        out = FixedQuantize(
            scale=1.0, zero_point=0.0, dtype=IntType(bits=8, signed=True)
        )(input)

        return out

    executor = BasicExecutor(network(input))
    executor.initialize()

    real_input = torch.rand(shape)
    output = executor.forward(real_input)

    # Convert the model to ONNX
    executor.eval()

    torch.onnx.export(executor, real_input, "test.onnx", verbose=True)

    # registry = torch.onnx.OnnxRegistry()

    # onnx_program = torch.onnx.dynamo_export(executor, real_input, export_options = torch.onnx.ExportOptions(op_level_debug=True, diagnostic_options=torch.onnx.DiagnosticOptions(verbosity_level=logging.DEBUG))).save('fixed_quantize.onnx')
