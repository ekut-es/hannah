import torch
import torch.fx as fx

from ...dataflow import DataFlowGraph
from ..core import Backend
from .dataflow import DataFlowModule


class _TorchBackend(Backend):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "torch"

    def build(self, dataflow_graph: DataFlowGraph) -> DataFlowModule:
        root = {}
        graph = fx.Graph()

        print()
        print("hierarchy:")
        for key, value in dataflow_graph.hierarchy():
            print("k:", key, "v:", value)

        for tensor, id in dataflow_graph.tensors.items():
            torch_tensor = torch.zeros(1, 1)
            root[str(id)] = torch_tensor

        module = DataFlowModule(root, graph)

        return module


TorchBackend = _TorchBackend()
