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
from abc import ABCMeta, abstractmethod
from typing import List, NamedTuple, Sequence

from ..expressions.placeholder import IntRange, UndefinedFloat, UndefinedInt
from ..hardware_description.memory_type import MemoryType
from ..parameters.parametrize import parametrize
from .registry import devices

logger = logging.getLogger(__name__)

Constraint = any
DataFlowGraph = any


class TargetOp(NamedTuple):
    name: str
    graph: DataFlowGraph
    constraints: Sequence[Constraint]

class DeviceMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace, /, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if not hasattr(cls, "name"):
            cls.name = name
            
        devices.register(cls)
        return cls

class Device(metaclass=DeviceMeta):
    name: str = ""
    description: str = ""
    _ops: List[TargetOp] 
    _memories: List[MemoryType]

    def __init__(self, name: str = "", description: str = "") -> None:
        super().__init__()
        if not name:
            name = "unnamed_device"
            logger.warning(
                "Unnamed device created. Please provide a name for the device."
            )

        self.name = name
        self.description = description
        self._ops: List = []
        self._memories = []

    def add_op(
        self,
        name: str,
        graph: DataFlowGraph,
        constraints: Sequence[Constraint],
    ) -> None:
        """Adds an operation to the device."""

        self._ops.append(TargetOp(name, graph, constraints))

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
