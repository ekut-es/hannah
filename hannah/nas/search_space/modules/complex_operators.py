import hannah.nas.search_space.modules.primitive_operators as ops
from torch.nn import Sequential, Module
from collections import OrderedDict
from hannah.nas.search_space.utils import get_same_padding


# From ProxylessNAS
class MBInvertedConvLayer(Module):
    """
    This layer is introduced in section 4.2 in the paper https://arxiv.org/pdf/1812.00332.pdf
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, mid_channels=None):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = Sequential()
        else:
            self.inverted_bottleneck = Sequential(OrderedDict([
                ('conv', ops.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', ops.BatchNorm2d(feature_dim)),
                ('act', ops.ReLU6(inplace=True)),
            ]))

        pad = get_same_padding(self.kernel_size)
        self.depth_conv = Sequential(OrderedDict([
            ('conv', ops.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)),
            ('bn', ops.BatchNorm2d(feature_dim)),
            ('act', ops.ReLU6(inplace=True)),
        ]))

        self.point_linear = Sequential(OrderedDict([
            ('conv', ops.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn', ops.BatchNorm2d(out_channels)),
        ]))

    def forward(self, x):
        x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x
