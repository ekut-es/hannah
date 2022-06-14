from tokenize import group
from typing import List
import torch.nn as nn
import torch.nn.functional as nnf
import logging
import math
import torch

from .elasticBase import ElasticBase1d
from ..utilities import (
    conv1d_get_padding,
    adjust_weights_for_grouping
)
from .elasticchannelhelper import SequenceDiscovery
from .elasticBatchnorm import ElasticWidthBatchnorm1d
from .elasticLinear import ElasticPermissiveReLU


class ElasticConv1d(ElasticBase1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        dilation_sizes: List[int],
        groups: List[int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        ElasticBase1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            padding=padding,
            dilation_sizes=dilation_sizes,
            groups=groups,
            bias=bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        # return self.get_basic_conv1d().forward(input)  # for validaing assembled module
        # get the kernel for the current index
        kernel, bias = self.get_kernel()
        dilation = self.get_dilation_size()
        # get padding for the size of the kernel

        padding = conv1d_get_padding(
            self.kernel_sizes[self.target_kernel_index], dilation
        )
        grouping = self.get_group_size()
        # MR 23123
        # adjust the kernel if grouping is done
        if(grouping > 1):
            # kernel_a = adjust_weights_for_grouping(kernel, 2)
            kernel_a = adjust_weights_for_grouping(kernel, grouping)
        else:
            kernel_a = kernel

        return nnf.conv1d(input, kernel_a, bias, self.stride, padding, dilation, grouping)

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        dilation = self.get_dilation_size()
        ##
        grouping = self.get_group_size()
        # if(grouping > 1):
        #     kernel = adjust_weights_for_grouping(kernel, grouping)
        ##
        padding = conv1d_get_padding(kernel_size, dilation)
        new_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=dilation,
            bias=False,
            groups=1
        )
        new_conv.weight.data = kernel
        if bias is not None:
            new_conv.bias = bias

        # print("\nassembled a basic conv from elastic kernel!")
        return new_conv


class ElasticConvReLu1d(ElasticBase1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        dilation_sizes: List[int],
        groups: List[int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        ElasticBase1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            padding=padding,
            dilation_sizes=dilation_sizes,
            groups=groups,
            bias=bias,
        )
        self.relu = ElasticPermissiveReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        # return self.get_basic_conv1d().forward(input)  # for validaing assembled module
        # get the kernel for the current index
        kernel, bias = self.get_kernel()
        dilation = self.get_dilation_size()
        # get padding for the size of the kernel
        padding = conv1d_get_padding(
            self.kernel_sizes[self.target_kernel_index], dilation
        )

        grouping = self.get_group_size()
        # MR 23123
        if(grouping > 1):
            kernel_a = adjust_weights_for_grouping(kernel, grouping)
        else:
            kernel_a = kernel

        return self.relu(
            nnf.conv1d(input, kernel_a, bias, self.stride, padding, dilation,  grouping)
        )

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        dilation = self.get_dilation_size()
        grouping = self.get_group_size()

        # if(grouping > 1):
        #     kernel = adjust_weights_for_grouping(kernel, grouping)

        padding = conv1d_get_padding(kernel_size, dilation)
        new_conv = ConvRelu1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            dilation_sizes=dilation,
            bias=False,
            groups=1
        )
        new_conv.weight.data = kernel
        if bias is not None:
            new_conv.bias = bias

        # print("\nassembled a basic conv from elastic kernel!")
        return new_conv


class ElasticConvBn1d(ElasticConv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        dilation_sizes: List[int],
        groups: List[int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        track_running_stats=False,
    ):
        ElasticBase1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            padding=padding,
            dilation_sizes=dilation_sizes,
            groups=groups,
            bias=bias,
        )
        self.bn = ElasticWidthBatchnorm1d(out_channels, track_running_stats)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        # return self.get_basic_conv1d().forward(input)  # for validaing assembled module
        dilation = self.get_dilation_size()
        # get padding for the size of the kernel
        self.padding = conv1d_get_padding(
            self.kernel_sizes[self.target_kernel_index], dilation
        )

        return self.bn(super(ElasticConvBn1d, self).forward(input))

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        dilation = self.get_dilation_size()
        grouping = self.get_group_size()

        # if(grouping > 1):
        #     kernel = adjust_weights_for_grouping(kernel, grouping)

        padding = conv1d_get_padding(kernel_size, dilation)
        new_conv = ConvBn1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=dilation,
            bias=False,
            groups=1
        )
        tmp_bn = self.bn.get_basic_batchnorm1d()

        new_conv.weight.data = kernel
        new_conv.bias = bias

        new_conv.bn.num_features = tmp_bn.num_features
        new_conv.bn.weight = tmp_bn.weight
        new_conv.bn.bias = tmp_bn.bias
        new_conv.bn.running_var = tmp_bn.running_var
        new_conv.bn.running_mean = tmp_bn.running_mean
        new_conv.bn.num_batches_tracked = self.bn.num_batches_tracked

        # print("\nassembled a basic conv from elastic kernel!")
        return new_conv


class ElasticConvBnReLu1d(ElasticConvBn1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        dilation_sizes: List[int],
        groups: List[int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        track_running_stats=False,
    ):
        ElasticConvBn1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            padding=padding,
            dilation_sizes=dilation_sizes,
            groups=groups,
            bias=bias,
        )

        self.relu = ElasticPermissiveReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        return self.relu(super(ElasticConvBnReLu1d, self).forward(input))

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        dilation = self.get_dilation_size()
        grouping = self.get_group_size()

        # if(grouping > 1):
        #     kernel = adjust_weights_for_grouping(kernel, grouping)

        padding = conv1d_get_padding(kernel_size, dilation)
        new_conv = ConvBnReLu1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=dilation,
            bias=False,
            groups=1
        )
        logging.info(f"Groups: {grouping}")
        tmp_bn = self.bn.get_basic_batchnorm1d()

        new_conv.weight.data = kernel
        new_conv.bias = bias

        new_conv.bn.num_features = tmp_bn.num_features
        new_conv.bn.weight = tmp_bn.weight
        new_conv.bn.bias = tmp_bn.bias
        new_conv.bn.running_var = tmp_bn.running_var
        new_conv.bn.running_mean = tmp_bn.running_mean
        new_conv.bn.num_batches_tracked = self.bn.num_batches_tracked

        # print("\nassembled a basic conv from elastic kernel!")
        return new_conv


class ConvRelu1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation_sizes: List[int],
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        track_running_stats=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation_sizes,
            groups=1,
            # groups=1
            bias=bias,
        )
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        if(self.groups == 1):
            return self.relu(super(ConvRelu1d, self).forward(input))

        #  MR 20220614 TODO hier weitermachen
        logging.info(f"Groups in forward: {self.groups}")
        full_kernel = torch.ones(self.weight.shape, device=self.weight.device)
        full_kernel.copy_(self.weight)
        if(self.groups > 1):
            self.weight = nn.Parameter(adjust_weights_for_grouping(self.weight, self.groups))
        tensor = self.relu(super(ConvRelu1d, self).forward(input))
        self.weight = nn.Parameter(full_kernel)
        return tensor


class ConvBn1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        track_running_stats=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels, track_running_stats=track_running_stats)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        if(self.groups == 1):
            return self.bn(super(ConvBn1d, self).forward(input))

        logging.info(f"Groups in forward: {self.groups}")
        full_kernel = torch.ones(self.weight.shape, device=self.weight.device)
        full_kernel.copy_(self.weight)
        if(self.groups > 1):
            self.weight = nn.Parameter(adjust_weights_for_grouping(self.weight, self.groups))
        tensor = self.bn(super(ConvBn1d, self).forward(input))
        self.weight = nn.Parameter(full_kernel)
        return tensor


class ConvBnReLu1d(ConvBn1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        track_running_stats=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels, track_running_stats=track_running_stats)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        if(self.groups == 1):
            return self.relu(super(ConvBnReLu1d, self).forward(input))

        logging.info(f"Groups in forward: {self.groups}")

        full_kernel = torch.ones(self.weight.shape, device=self.weight.device)
        full_kernel.copy_(self.weight)
        if(self.groups > 1):
            self.weight = nn.Parameter(adjust_weights_for_grouping(self.weight, self.groups))
        tensor = self.relu(super(ConvBnReLu1d, self).forward(input))
        self.weight = nn.Parameter(full_kernel)
        return tensor
