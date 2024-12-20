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

import logging
from typing import List, Sequence

from hannah.nas.core.expression import Expression
from hannah.nas.expressions.placeholder import UndefinedInt
from hannah.nas.functional_operators.op import BaseNode, Op, Tensor
from hannah.nas.parameters import CategoricalParameter

from .base import DescriptionBackend
from .utils import all_nodes


class MatchedRegion:
    """A matched region of a hannah pattern."""

    def __init__(self, pattern, nodes):
        self.pattern = pattern
        self.nodes = nodes

    def __str__(self):
        return f"MatchedRegion(pattern={self.pattern.name}, nodes={[n.id for n in self.nodes]})"


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
            if match:
                match = MatchedRegion(self._pattern, match)
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
            print("matching:", current_node.id, current_pattern.id)

            # Dispatch over the different types of nodes
            if isinstance(current_node, Op):
                matches = self._match_op(current_node, current_pattern)
                if not matches:
                    return []
            elif isinstance(current_node, Tensor):
                matches = self._match_tensor(current_node, current_pattern)
                if not matches:
                    return []
            partial_match.append(current_node)

            print("partial match:", [n.id for n in partial_match])

            if current_pattern.operands:
                if len(current_node.operands) != len(current_pattern.operands):
                    return []

                for operand, pattern_operand in zip(
                    current_node.operands, current_pattern.operands
                ):
                    worklist.append((operand, pattern_operand))

        return partial_match

    def _match_op(self, node: Op, pattern: Op) -> bool:
        """Matches an op node."""

        # Iterate over attributes of op node
        for attr in node.__dict__:
            if attr.startswith("_"):
                continue

            if attr == "operands":
                continue

            if attr == "users":
                continue

            # FIXME: use regex match here
            if attr == "id":
                continue
            if attr == "name":
                continue
            if attr == "scope":
                continue

            # print("Matching attributes: ", attr)
            if not self._is_subset(getattr(node, attr), getattr(pattern, attr)):
                print("Failed to match attributes: ", attr)
                return False

        return True

    def _match_tensor(self, node: BaseNode, pattern: Tensor) -> bool:
        """Matches a tensor pattern, the node pattern can still be an op.

        In the case of matching against an op pattern we only check the shape.
        """

        if isinstance(pattern, Op):
            logging.critical(
                "Matching a tensor against an op pattern is not implemented correctly, shapes might not match"
            )
            return True

        if len(node.shape()) != len(pattern.shape()):
            return False
        else:
            for dim, pattern_dim in zip(node.shape(), pattern.shape()):
                if not self._is_subset(dim, pattern_dim):
                    return False

        return True

    def _is_subset(self, node_attr, pattern_attr):
        """Checks if a node attribute is a subset of the pattern attribute."""

        print("Matching attribute sets: ", node_attr, pattern_attr)

        if isinstance(node_attr, CategoricalParameter):
            node_set = set(node_attr.choices)
            if isinstance(pattern_attr, UndefinedInt):
                return True
            elif hasattr(pattern_attr, "values"):
                pattern_set = set(pattern_attr.values)
            elif isinstance(pattern_attr, Expression):
                logging.critical(
                    "Matching a categorical parameter against an expression is not implemented correctly"
                )
            elif isinstance(pattern_attr, int):
                pattern_set = set([pattern_attr])

            if not (node_set & pattern_set):
                return False
            else:
                # FIXME: this is a hack to make sure that the choices are only the ones in the pattern
                # FIXME: this can result in unexpected behaviour if the current_value is not set to one of the restricted values
                node_attr.choices = [x for x in node_attr.choices if x in pattern_set]
        elif isinstance(node_attr, Expression):
            return True
        elif isinstance(node_attr, int):
            if isinstance(pattern_attr, int):
                if node_attr != pattern_attr:
                    return False
            elif isinstance(pattern_attr, UndefinedInt):
                return True
            else:
                logging.critical(
                    f"Matching an int against an expression of type: {type(pattern_attr)} is not implemented correctly"
                )
                return True

            return True

        return True


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
            self._matched_regions.extend(pattern.match(nodes))

        return self._matched_regions

    @property
    def matches(self) -> Sequence[MatchedRegion]:
        """Returns the matched regions."""
        return self._matched_regions


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
