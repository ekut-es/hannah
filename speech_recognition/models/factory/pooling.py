import torch
from torch import nn
from ..utils import next_power_of2


class ApproximateGlobalAveragePooling1D(nn.Module):
    """A global average pooling layer, that divides by the next power of 2 instead of true number of elements"""

    def __init__(self, size):
        super().__init__()

        self.size = size
        self.divisor = next_power_of2(size)

    def forward(self, x):
        x = torch.sum(x, dim=2, keepdim=True)
        x = x / self.divisor

        return x
