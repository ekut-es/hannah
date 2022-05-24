from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ...dataflow.dataflow_graph import DataFlowGraph

DFType = TypeVar("DFType")


class Backend(ABC, Generic[DFType]):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def build(self, dataflow_graph: DataFlowGraph) -> DFType:
        pass
