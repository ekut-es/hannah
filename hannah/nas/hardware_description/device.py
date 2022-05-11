from typing import List
from hannah.nas.dataflow.dataflow_graph import DataFlowGraph
from hannah.nas.parameters import parametrize
from abc import ABC, abstractmethod

class Device(ABC):


    def ops(self) -> List[DataFlowGraph]:
        ...

    def memories(self) -> List[MemoryType]:
        ...


@parametrize
class Ultratrail(Device):
