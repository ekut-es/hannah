import torch
from torch import nn

from ..utils import next_power_of2


class ApproximateGlobalAveragePooling1D(nn.Module):
    """A global average pooling layer, that divides by the next power of 2 instead of true number of elements"""

    def __init__(self, size, qconfig=None):
        super().__init__()

        self.size = size
        self.divisor = next_power_of2(size)
        self.qconfig = qconfig
        if qconfig:
            self.activation_post_process = qconfig.activation()

    def forward(self, x):
        x = torch.sum(x, dim=2, keepdim=True)
        x = x / self.divisor

        if hasattr(self, "activation_post_process"):
            x = self.activation_post_process(x)

        return x


class ApproximateGlobalAveragePooling2D(nn.Module):
    """A global average pooling layer, that divides by the next power of 2 instead of true number of elements"""

    def __init__(self, size, qconfig=None):
        super().__init__()

        self.size = size
        self.divisor = next_power_of2(size)
        self.qconfig = qconfig
        if qconfig:
            self.activation_post_process = qconfig.activation()

    def forward(self, x):
        x = torch.sum(x, dim=[2, 3], keepdim=True)
        x = x / self.divisor

        if hasattr(self, "activation_post_process"):
            x = self.activation_post_process(x)

        return x
