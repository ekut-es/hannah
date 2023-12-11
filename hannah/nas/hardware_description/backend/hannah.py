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
"""
Translates target patterns to be applied on the hannah neural network search space descriptions.
"""

from typing import List

from hannah.nas.functional_operators.op import BaseNode, Op, Tensor

from .base import DescriptionBackend
from .utils import all_nodes


class HannahPattern:
    """Pattern for hannah search space descriptions."""

    def __init__(self, name, pattern: BaseNode, condition):
        self.name = name
        self._pattern = pattern
        self._condition = condition

    def match(self, nodes):
        """Matches the pattern on a search space."""

        matches = []

        for node in nodes:
            match = self._match_node(node)
            if matches:
                matches.append(match)
        return matches

    def _match_node(self, node) -> List[BaseNode]:
        """Matches the pattern on a single node.
        Returns a list of nodes in the matched subexpression in post-order.
        """
        partial_match = []

        worklist = [(node, self._pattern)]
        while worklist:
            current_node, current_pattern = worklist.pop()

            # Dispatch over the different types of nodes
            if isinstance(current_node, Op):
                matches = self._match_op(current_node, current_pattern)
                if not matches:
                    return []
            elif isinstance(current_node, Tensor):
                matches = self._match_tensor(current_node, current_pattern)
                if not matches:
                    return []

            if len(current_node.operands) != len(current_pattern.operands):
                return []

            partial_match.append(current_node)

            for operand, pattern_operand in zip(
                current_node.operands, current_pattern.operands
            ):
                worklist.append(operand, pattern_operand)

        return None


class HannahMatcher:
    """Matcher for hannah patterns."""

    def __init__(self, name: str, patterns: List[HannahPattern]):
        self.name = name
        self._patterns = patterns
        self._matched_regions = []

    def run(self, search_space):
        """Runs the matcher on a search space."""

        nodes = all_nodes(search_space)
        for pattern in self._patterns:
            self._matched_regions.append(pattern.match(nodes))


class HannahBackend(DescriptionBackend):
    """Generator for hannah data sheets from target devices."""

    def __init__(self):
        super().__init__()

    def generate(self, device) -> HannahMatcher:
        """Generates a hannah description from a target device meta model."""

        patterns = []
        for name, op, cond in device.ops:
            patterns.append(HannahPattern(name, op, cond))

        backend = HannahMatcher(device.name, patterns)

        return backend
