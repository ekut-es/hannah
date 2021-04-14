import torch.fx

from .qat import (
    Conv1d,
    Conv2d,
    ConvBn1d,
    ConvBn2d,
    ConvBnReLU1d,
    ConvBnReLU2d,
    ConvReLU1d,
    ConvReLU2d,
)


class QuantizationTracer(torch.fx.Tracer):

    LEAF_MODULES = [
        Conv1d,
        Conv2d,
        ConvBn1d,
        ConvBn2d,
        ConvBnReLU1d,
        ConvBnReLU2d,
        ConvReLU1d,
        ConvReLU2d,
    ]

    def is_leaf_module(self, module, module_qualified_name):
        for leaf_cls in self.LEAF_MODULES:
            if isinstance(module, leaf_cls):
                return True

        return super().is_leaf_module(module, module_qualified_name)


class RelayConverter(torch.fx.Interpreter):
    def run_node(self, n):
        result = super().run_node(n)

        breakpoint()

        return result

    def propagate(self, *args):
        return super().run(*args)
