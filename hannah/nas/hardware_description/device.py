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
from abc import ABC, abstractmethod
from typing import List, Sequence

from ..dataflow.dataflow_graph import DataFlowGraph, dataflow
from ..dataflow.op_type import OpType
from ..expressions.placeholder import IntRange, UndefinedFloat, UndefinedInt
from ..hardware_description.memory_type import MemoryType
from ..ops import (
    add,
    avg_pool,
    axis,
    broadcast,
    int_t,
    optional,
    quantization,
    relu,
    requantize,
    tensor,
)
from ..parameters.parametrize import parametrize


class TargetOp(NamedTuple):
    name: str
    graph: DataFlowGraph
    constraints: Sequence[Constraint]


class Device(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._ops: List = []
        self._memories = []

    @property
    def ops(self) -> Sequence[TargetOp]:
        return self._ops

    @property
    def memories(self) -> List[MemoryType]:
        return self._memories

    def __str__(self):
        res = self.__class__.__name__ + ":\n"
        res += "Ops:\n"
        for op in self.ops:
            res += str(op) + "\n"
        for memory in self.memories:
            res += str(memory) + "\n"
        return res
