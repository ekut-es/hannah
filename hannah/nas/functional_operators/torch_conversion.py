import torch
from hannah.nas.functional_operators.lazy import lazy
from hannah.nas.functional_operators.op import Op, Tensor
from hannah.nas.functional_operators.operators import Conv2d, Relu, Linear, BatchNorm, AdaptiveAvgPooling, Add
from hannah.nas.functional_operators.executor import BasicExecutor
from hannah.models.embedded_vision_net.models import search_space
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False)

    def forward(self, x):
        out = torch.flatten(x, start_dim=1)
        out = self.linear(out)
        return out


class TorchConverter(BasicExecutor):
    def __init__(self, search_space) -> None:
        super().__init__(search_space)
        self.conversions = {Conv2d: self.convert_conv2d,
                            Relu: self.convert_relu,
                            Linear: self.convert_linear,
                            BatchNorm: self.convert_batch_norm,
                            AdaptiveAvgPooling: self.convert_adaptive_avg_pool}
        self.mods = nn.ModuleDict()

    def convert_conv2d(self, node):
        in_channels = lazy(node.in_channels)
        out_channels = lazy(node.out_channels)
        kernel_size = lazy(node.kernel_size)
        stride = lazy(node.stride)
        padding = lazy(node.padding)
        dilation = lazy(node.dilation)
        groups = lazy(node.groups)

        return nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=False)  # FIXME: Add Bias to functional convs

    def convert_relu(self, node):
        return nn.ReLU()

    def convert_linear(self, node):
        in_features = lazy(node.in_features)
        out_features = lazy(node.out_features)
        return nn.Linear(in_features, out_features)

    def convert_batch_norm(self, node):
        num_features = lazy(node.operands[0].shape()[1])
        return nn.BatchNorm2d(num_features=num_features)

    def convert_max_pool(self, node):
        pass

    def convert_avg_pool(self, node):
        pass

    def convert_adaptive_avg_pool(self, node):
        output_size = lazy(node.output_size)
        return nn.AdaptiveAvgPool2d(output_size=output_size)

    def convert(self):
        self.find_execution_order()
        to_remove = []
        for node_name in self.nodes:
            node = self.node_dict[node_name]
            if type(node) in self.conversions:
                op = self.conversions[type(node)](node)
                self.mods[node_name.replace(".", "_")] = op
                remove_operands = [operand for operand in self.forward_dict[node_name] if not isinstance(self.node_dict[operand], Op) and not operand == "input"]
                for operand in remove_operands:
                    self.forward_dict[node_name].remove(operand)
            elif not isinstance(node, Op):
                to_remove.append(node_name)

        for node_name in to_remove:
            self.nodes.remove(node_name)
            del self.forward_dict[node_name]
        print()

    def forward(self, x):
        # FIXME: Remove obsolete entries in out
        out = {'input': x}
        for node in self.nodes:
            node_name = node.replace(".", "_")
            operands = [out[n] for n in self.forward_dict[node]]

            if node_name in self.mods:
                if isinstance(self.mods[node_name], nn.Linear):
                    operands = [torch.flatten(operands[0], start_dim=1)]
                out[node] = self.mods[node_name](*operands)
            elif isinstance(self.node_dict[node], Add):
                out[node] = torch.add(*operands)
        return out[node]


if __name__ == '__main__':
    input = Tensor(name="input", shape=(32, 3, 32, 32), axis=("N", "C", "H", "W"))
    space = search_space(name="evn", input=input, num_classes=10)
    # space.sample()
    converter = TorchConverter(space)
    converter.convert()
    x = torch.randn(input.shape())
    converter.forward(x)
