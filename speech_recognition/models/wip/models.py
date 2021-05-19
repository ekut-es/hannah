import torch.nn as nn
import numpy as np
# import torch

# from ..utils import ConfigType, SerializableModule

# class WIPModel(SerializableModule):


class WIPModel(nn.Module):
    """Partial clone of OFA: DynamicConv2d | github.com/mit-han-lab/once-for-all"""

    def __init__(self, config):
        super().__init__()

        self.max_depth = config["max_depth"]
        self.max_kernel_size = config["max_kernel_size"]
        self.active_out_channel = config["max_out_channels"]
        self.config = config

        self.active_depth = self.max_depth

        # from convolution result dims
        self.pool = nn.AvgPool1d(101)
        self.flatten = nn.Flatten(1)
        self.linear = nn.Linear(40, 12)  # from channel count at output

        self.convLayers = nn.ModuleList([])

        for i in range(1, self.max_depth):
            self.convLayers.append(
                nn.Conv1d(
                    # in_channels = config["max_in_channels"],
                    # out_channels = config["max_out_channels"],
                    in_channels=40,  # temporary hardcode, should be from config
                    out_channels=40,
                    kernel_size=config["kernel_size"],
                    stride=config["stride"],
                    dilation=config["dilation"],
                    bias=config["bias"],
                    padding=4  # from kernel size
                )
            )

    # filter from DynamicConv2d
    def get_active_filter(self, out_channel, in_channel):
        # out_channels, in_channels/groups, kernel_size[0], kernel_size[1]
        return self.conv.weight[:out_channel, :in_channel, :]

    # adapted filter from DynamicSeparableConv2d
    """
    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        start, end = sub_filter_start_end(self.max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        return filters
    """

    # def forward(self, x, out_channel=None):
    def forward(self, x):
        """
        out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.get_active_filter(out_channel, in_channel).contiguous()
        padding = get_padding(self.conv.kernel_size)
        result = nn.functional.conv1d(x, filters, self.conv.bias, self.conv.stride, padding, self.conv.dilation)
        """

        for layer in self.convLayers[:self.active_depth]:
            x = layer(x)

        result = x
        # print(np.shape(result))
        result = self.pool(result)
        result = self.flatten(result)
        result = self.linear(result)
        # print(np.shape(result))

        return result
        # return nn.functional.conv2d(x, self.conv.weight, None, self.conv.stride, padding, self.conv.dilation, 1)

    def sample_active_subnet(self):
        self.active_depth = np.random.randint(1, self.max_depth)
        # print("Picked active depth: ", self.active_depth)


def get_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        p1 = get_padding(kernel_size[0])
        # p2 = get_padding(kernel_size[1])
        # return p1, p2
        return p1
    return kernel_size // 2


"""
def sub_filter_start_end(kernel_size, sub_kernel_size):
center = kernel_size // 2
dev = sub_kernel_size // 2
start, end = center - dev, center + dev + 1
#assert end - start == sub_kernel_size
return start, end
"""
