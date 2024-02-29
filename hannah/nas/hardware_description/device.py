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
from abc import ABCMeta, abstractmethod
from typing import List, NamedTuple, Sequence

from hannah.nas.core.parametrized import is_parametrized

from ..expressions.placeholder import IntRange, UndefinedFloat, UndefinedInt
from ..functional_operators.utils.visit import reverse_post_order
from ..hardware_description.memory_type import CouplingType, ManagementType, MemoryType
from ..parameters.parametrize import parametrize
from .registry import devices

logger = logging.getLogger(__name__)

Constraint = any
DataFlowGraph = any


class TargetOp(NamedTuple):
    name: str
    graph: DataFlowGraph
    constraints: Sequence[Constraint]

    def markdown(self) -> str:
        res = "### " + self.name + "\n"

        ids = {}
        res += "\nGraph:\n"
        res += "```mlir\n"
        for num, node in enumerate(reverse_post_order(self.graph)):
            node_list = []
            ids[node.id] = f"%{node.id}_{num}"
            for operand in node.operands:
                node_list.append(ids[operand.id])

            for attr, value in node.attributes().items():

                def value_to_str(value):
                    if isinstance(value, IntRange):
                        value_str = f"{value.min}..{value.max}"
                    elif isinstance(value, UndefinedInt):
                        value_str = f"?{str(value.id)}"
                    elif isinstance(value, UndefinedFloat):
                        value_str = f"?{str(value.id)}"
                    elif isinstance(value, Sequence) and not isinstance(value, str):
                        value_str = (
                            "[" + ", ".join(value_to_str(x) for x in value) + "]"
                        )
                    else:
                        value_str = str(value)
                    return value_str

                value_str = value_to_str(value)

                node_list.append(f"{attr}={value_str}")

            node_str = f"{node.name}({', '.join(node_list)})"

            res += f"%{node.id}_{num} = {node_str}\n"
        res += "```\n"

        res += "\nConstraints:\n"
        ids = {}
        for constraint in self.constraints:
            res += "- " + str(constraint) + "\n"

        return res


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

    def __init__(self) -> None:
        super().__init__()

        self._ops: List[TargetOp] = []
        self._memories = []

    def add_memory(
        self,
        scope: str,
        size: int,
        latency: int,
        management: ManagementType = ManagementType.EXPLICIT,
        coupling: CouplingType = CouplingType.COUPLED,
        read_bw: int = 10,
        write_bw: int = 10,
        read_energy: int = 10,
        write_energy: int = 10,
        idle_energy: int = 10,
        area: int = 10,
        read_port: int = 10,
        write_port: int = 10,
        rw_port: int = 10,
    ) -> None:
        self._memories.append(
            MemoryType(
                scope=scope,
                size=size,
                latency=latency,
                management=management,
                coupling=coupling,
                read_bw=read_bw,
                write_bw=write_bw,
                read_energy=read_energy,
                write_energy=write_energy,
                idle_energy=idle_energy,
                area=area,
                read_port=read_port,
                write_port=write_port,
                rw_port=rw_port,
            )
        )

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
