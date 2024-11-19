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
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from ...functional_operators.op import BaseNode


def as_nx_graph(op: "BaseNode") -> nx.DiGraph:
    """Returns a networkx graph representation of the operator graph"""
    graph = nx.DiGraph()

    visited = set()
    worklist = [op]
    while len(worklist) > 0:
        current = worklist.pop()
        if current in visited:
            continue
        visited.add(current)
        for operand in current.operands:
            worklist.append(operand)
            graph.add_edge(operand.id, current.id)

    return graph


def as_string(op: "BaseNode") -> str:
    """Returns a string representation of the operator graph"""
    return nx.write_network_text(as_nx_graph(op))


def as_dot(op: "BaseNode") -> str:
    """Returns a dot representation of the operator graph"""
    return nx.nx_pydot.to_pydot(as_nx_graph(op)).to_string()


def write_png(op: "BaseNode", filename: str) -> None:
    """Writes a png file of the operator graph"""
    nx.nx_pydot.to_pydot(as_nx_graph(op)).write_png(filename)


def write_pdf(op: "BaseNode", filename: str) -> None:
    """Writes a pdf file of the operator graph"""
    nx.nx_pydot.to_pydot(as_nx_graph(op)).write_pdf(filename)
