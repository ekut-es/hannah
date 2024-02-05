from _typeshed import SupportsWrite
from abc import ABCMeta
from .op import Op


class Quantize(Op):
    def __init__(self, input, output, scale, zero_point):
        super().__init__(input, output)
        self.scale = scale
        self.zero_point = zero_point

    def shape_fun(self):
        return self.operands[0].shape()

    def _forward_implementation(self, x):
        return x
    
    