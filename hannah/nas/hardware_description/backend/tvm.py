#
# Copyright (c) 2023 Hannah contributors.
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

from ...functional_operators.operators import ChoiceOp, Op, Tensor
from ..device import Device, TargetOp
from .base import DescriptionBackend

logger = logging.getLogger(__name__)


_TVM_OP_TABLE = {
    "Conv1d": "nn.conv1d",
    "Conv2d": "nn.conv2d",
    "Add": "add",
    "Relu": "nn.relu",
    "Linear": "nn.linear",
    "MaxPool2d": "nn.max_pool2d",
}


class TVMBackend(DescriptionBackend):
    def __init__(self):
        self._device = None
        self._pattern_table = ""

    def generate(self, device: Device):
        self._device = device
        self._pattern_table = "from tvm import relay\n"
        self._pattern_table += "from tvm.relay import Expr\n"
        self._pattern_table += (
            "from tvm.relay.dataflow_pattern import wildcard, is_op\n"
        )
        self._pattern_table += (
            "from tvm.relay.op.contrib.register import register_pattern_table\n"
        )
        self._pattern_table += "\n\n"

        for op in device.ops:
            self._pattern_table += self._handle_graph(op)
            self._pattern_table += "\n\n"

        self._pattern_table += f'@register_pattern_table("{self._device.name}")\n'
        self._pattern_table += "def pattern_table():\n"
        self._pattern_table += "    return [\n"

        for op in device.ops:
            self._pattern_table += (
                f"        ({op.name}.name, {op.name}.pattern(), {op.name}.check),\n"
            )

        self._pattern_table += "    ]\n"

        print(self._pattern_table)

        # compiled = compile(self._pattern_table, "", "exec")
        # mod = exec(compiled, globals(), locals())

        return self._pattern_table

    def _handle_graph(self, op: TargetOp) -> str:
        worklist = [op.graph]
        ops = []

        while worklist:
            current_op = worklist.pop()
            if current_op in ops:
                continue

            ops.append(current_op)

            for child in current_op.operands:
                worklist.append(child)

        result = f"class  {op.name}:\n"
        result += f'    name = "{self._device.name}.{op.name}" \n'

        # generate Pattern
        result += "    @classmethod\n"
        result += "    def pattern(cls):\n"

        id_table = {}
        for num, op in enumerate(reversed(ops)):
            op_id = f"{op.name}_{num}".lower()

            id_table[op] = op_id

            if isinstance(op, Tensor):
                matcher = "wildcard()"  # FIXME: handle consts
            elif isinstance(op, ChoiceOp):
                matcher = "||".join([f"{id_table[o]}" for o in op.options])
            elif isinstance(op, Op):
                if op.name not in _TVM_OP_TABLE:
                    raise NotImplementedError(f"Unsupported op {op.name}")

                tvm_name = _TVM_OP_TABLE[op.name]
                matcher = f'is_op("{tvm_name}")('
                for operand in op.operands[:-1]:
                    matcher += f"{id_table[operand]}, "
                matcher += f"{id_table[op.operands[-1]]})"
            else:
                raise NotImplementedError(f"Unsupported op {op.name}")

            result += f"        {op_id} = {matcher}\n"

        result += f"        return {id_table[ops[0]]}\n"

        # Generate Verification Code
        result += "    @classmethod\n"
        result += '    def check(cls, expr: "Expr") -> bool:\n'
        result += "        return True\n"

        return result
