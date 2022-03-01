import hannah.nas.search_space.modules.primitive_operators as p_ops
from hannah.nas.search_space.modules.complex_operators import MBInvertedConvLayer
from torch.nn import Module
import torch


class ZeroLayerModule(Module):
    def __init__(self, shortcut):
        super().__init__()
        self.shortcut = shortcut

    def forward(self, x, out):
        if torch.sum(torch.abs(out)).item() == 0:
            if x.size() == out.size():
                # is zero layer
                return x
        if self.shortcut is None:
            return out
        return out + self.shortcut(x)


class MobileInvertedResidualBlock(Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 expand_ratio=6,
                 mid_channels=None,
                 shortcut=True):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = MBInvertedConvLayer(in_channels,
                                                        out_channels,
                                                        kernel_size,
                                                        stride,
                                                        expand_ratio,
                                                        mid_channels)
        if shortcut:
            shortcut = p_ops.FactorizedReduce(in_channels, out_channels, stride)
        else:
            shortcut = None
        self.zero_layer_module = ZeroLayerModule(shortcut)

    def forward(self, x):
        out = self.mobile_inverted_conv(x)
        return self.zero_layer_module(x, out)


class Classifier(Module):
    def __init__(self, in_channels, last_channels, n_classes) -> None:
        super().__init__()
        self.feature_mix_layer = p_ops.Conv2d(in_channels=in_channels, out_channels=last_channels, kernel_size=1)
        self.relu = p_ops.ReLU6()
        self.pool = p_ops.AdaptiveAvgPool2d(1)
        self.linear = p_ops.Linear(last_channels, n_classes)

    def forward(self, x):
        out = self.feature_mix_layer(x)
        out = self.relu(out)
        out = self.pool(out).squeeze(-1).squeeze(-1)
        out = self.linear(out)
        return out
