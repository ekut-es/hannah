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
from hannah.nas.core.expression import Expression
from hannah.nas.core.parametrized import is_parametrized


def extract_parameter_from_expression(expression):
    assert isinstance(expression, Expression)
    queue = expression.get_children()
    params = []
    visited = expression.get_children()
    while queue:
        current = queue.pop(0)
        if is_parametrized(current):
            params.append(current)
        elif isinstance(current, Expression):
            children = current.get_children()
            for c in children:
                # the following if/else with c_ is just to include tuples and lists
                if isinstance(c, (tuple, list)):
                    c_ = c
                else:
                    c_ = [c]
                for x in c_:
                    if not any(
                        [x is v for v in visited]
                    ):  # Hack because EQCondition messes with classical "if x in list" syntax
                        queue.append(x)
                        visited.append(x)

    return params


def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result
