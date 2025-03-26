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
from hannah.nas.expressions.choice import SymbolicSequence
from hannah.nas.expressions.utils import extract_parameter_from_expression
from ..op import BaseNode
from hannah.nas.functional_operators.op import ChoiceOp
from hannah.nas.functional_operators.operators import Conv2d
from hannah.nas.parameters.parameters import Parameter
from hannah.nas.core.expression import Expression



def post_order(op: BaseNode):
    """Visits the operator graph in post order"""
    visited = set()
    worklist = [op]
    while len(worklist) > 0:
        current = worklist[-1]
        if current in visited:
            worklist.pop()
            continue
        visited.add(current)
        for operand in current.operands:
            if operand not in visited:
                worklist.append(operand)
        if current in worklist:
            worklist.remove(current)
        yield current


def reverse_post_order(op: BaseNode):
    """Visits the operator graph in reverse post order"""
    return reversed(list(post_order(op)))


def get_active_parameters(space, parametrization=None):
    if parametrization is None:
        parametrization = space.parametrization()

    queue = [space]
    visited = [space.id]
    active_params = {}

    while queue:
        node = queue.pop(0)
        # FIXME: This should be done in parametrize.py. This is just a hack
        if isinstance(node, Conv2d):
            node_attrs = {'out_channels': node.out_channels, "in_channels": node.in_channels, "kernel_size": node.kernel_size, "stride": node.stride, "dilation": node.dilation, "groups": node.groups}
            params = {}
            for k, p in node_attrs.items():
                if isinstance(p, Parameter):
                    params[p.name] = p
                elif isinstance(p, SymbolicSequence):
                    pass
                elif isinstance(p, Expression):
                    extracted_params = extract_parameter_from_expression(p)
                    for extracted in extracted_params:
                        params[extracted.name] = extracted

        else:
            params = node._PARAMETERS
        for k, p in params.items():
            if isinstance(p, Parameter):
                if p.id in parametrization:
                    active_params[p.id] = parametrization[p.id]
        for operand in node.operands:
            while isinstance(operand, ChoiceOp):
                for k, p in operand._PARAMETERS.items():
                    if isinstance(p, Parameter):
                        if p.id in parametrization:
                            active_params[p.id] = parametrization[p.id]
                if operand.switch.id in parametrization:
                    active_op_index = parametrization[operand.switch.id].evaluate()
                else:
                    active_op_index = operand.switch.evaluate()  # FIXME: TEST
                operand = operand.operands[active_op_index]
            if operand.id not in visited:
                queue.append(operand)
                visited.append(operand.id)
    return active_params
