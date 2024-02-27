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
from ..op import BaseNode


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
