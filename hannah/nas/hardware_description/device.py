from typing import List
from hannah.nas.dataflow.dataflow_graph import DataFlowGraph
from hannah.nas.hardware_description.memory_type import MemoryType
from hannah.nas.parameters import parametrize
from abc import ABC, abstractmethod


class Device(ABC):
    def ops(self) -> List[DataFlowGraph]:
        ...

    def memories(self) -> List[MemoryType]:
        ...


@parametrize
class Ultratrail(Device):
    def __init__(self,
                 weight_bits: int = 8,
                 bias_bits: int = 8,
                 activation_bits: int = 8,
                 accumulator_bits: int = 8,
                 max_weight_bits: int = 8) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.bias_bits = bias_bits
        self.activation_bits = activation_bits
        self.accumulator_bits = accumulator_bits
        self.max_weight_bits = max_weight_bits
