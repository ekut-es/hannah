import torch
import torch.fx as fx


class DataFlowModule(fx.GraphModule):
    def __init__(self, root, graph):
        super().__init__(root, graph, class_name=self.__class__.__name__)
