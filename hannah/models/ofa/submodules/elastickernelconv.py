from inspect import Parameter
import random
from tokenize import group
from typing import List
import torch.nn as nn
import torch.nn.functional as nnf
import copy
import logging
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from ..utilities import conv1d_get_padding
from .elasticBase import ElasticBase1d
from ..utilities import (
    adjust_weight_if_needed,
    conv1d_get_padding,
    adjust_weights_for_grouping,
    is_weight_adjusting_needed,
    pre_hook_forward
)
from .elasticBatchnorm import ElasticWidthBatchnorm1d
from .elasticLinear import ElasticPermissiveReLU


# MR 20220622
# TODO vereinheitlichen

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
        out_channel_sizes=None,
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
            out_channel_sizes=out_channel_sizes,
        )
        self.norm = False
        self.act = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
        # grouping_changed = grouping != self.last_grouping_param
        # if(grouping_changed and grouping > 1):
        #     # kernel_a = adjust_weights_for_grouping(kernel, 2)
        #     kernel = adjust_weights_for_grouping(kernel, grouping)
        kernel, _ = adjust_weight_if_needed(module=self, kernel=kernel, groups=grouping, in_place_adjustment=False)

        return nnf.conv1d(input, kernel, bias, self.stride, padding, dilation, grouping)

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        dilation = self.get_dilation_size()
        ##
        grouping = self.get_group_size()

        padding = conv1d_get_padding(kernel_size, dilation)

        #self.in_channels = kernel.size(1)
        #self.out_channels = kernel.size(0)
        self.set_in_and_out_channel(kernel)

        new_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=dilation,
            bias=False,
            groups=grouping
        )
        new_conv.last_grouping_param = self.groups

        if not hasattr(new_conv, 'id'):
            new_conv.id = "ElasticConv1d-" + str(random.randint(0, 1000)*2000)
            logging.debug(f"Validation id created: {new_conv.id} ; g={grouping}, w_before={kernel.shape}, ic={self.in_channels}")
        else:
            logging.debug("Validation id already present: {new_conv.id}")

        kernel, _ = adjust_weight_if_needed(module=new_conv, kernel=kernel, groups=new_conv.groups, in_place_adjustment=False)
        new_conv.weight.data = kernel
        if bias is not None:
            new_conv.bias = bias

        logging.debug(f"=====> id: {new_conv.id} ; g={grouping}, w_after={kernel.shape}, ic={self.in_channels}")
        # new_conv.forward = self.hook_forward
        # new_conv.register_forward_pre_hook(pre_hook_forward)
        # print("\nassembled a basic conv from elastic kernel!")
        return new_conv

    # def pre_hook_forward(self, module, input: torch.Tensor) -> torch.Tensor:
    #     """
    #         This Hook is called before the forward will be executed.
    #         TODO: maybe use that for normal conv evolution as well ?
    #     """
    #     grouping_changed = module.groups != module.first_grouping
    #     logging.info(f"Shape:{module.weight.shape} Groups:{module.groups} Group_First: {module.first_grouping} groups_changed:{grouping_changed} ic={module.in_channels}, oc={module.out_channels}")
    #     if grouping_changed and module.groups > 1:
    #         weight_adjustment_needed = is_weight_adjusting_needed(module.weight, module.in_channels, module.groups)
    #         if weight_adjustment_needed:
    #             module.weight = nn.Parameter(adjust_weights_for_grouping(module.weight, module.groups))
    #             logging.info(f"NOW Shape:{module.weight.shape} Groups:{module.groups} Group_First: {module.first_grouping} groups_changed:{grouping_changed} ic={module.in_channels}, oc={module.out_channels}")
        # kernel = self.weight.data
        # if(self.groups > 1):
        #     kernel = adjust_weights_for_grouping(self.weight.data, self.groups)


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
        out_channel_sizes=None,
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
            out_channel_sizes=out_channel_sizes,
        )
        self.relu = ElasticPermissiveReLU()
        self.norm = False
        self.act = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
        kernel, _ = adjust_weight_if_needed(module=self, kernel=kernel, groups=grouping, in_place_adjustment=False)

        return self.relu(
            nnf.conv1d(input, kernel, bias, self.stride, padding, dilation,  grouping)
        )

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        dilation = self.get_dilation_size()
        grouping = self.get_group_size()

        # if(grouping > 1):
        #     kernel = adjust_weights_for_grouping(kernel, grouping)
        #self.in_channels = kernel.size(1)
        #self.out_channels = kernel.size(0)
        self.set_in_and_out_channel(kernel)

        padding = conv1d_get_padding(kernel_size, dilation)
        new_conv = ConvRelu1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            dilation_sizes=dilation,
            bias=False,
            groups=grouping
        )
        new_conv.last_grouping_param = self.groups
        if not hasattr(new_conv, 'id'):
            new_conv.id = "ConvRelu1d-" + str(random.randint(0, 1000)*2000)
            logging.debug(f"Validation id created: {new_conv.id} ; g={grouping}, w_before={kernel.shape}, ic={self.in_channels}")
        else:
            logging.debug("Validation id already present: {new_conv.id}")

        kernel, _ = adjust_weight_if_needed(module=new_conv, kernel=kernel, groups=new_conv.groups, in_place_adjustment=False)
        logging.debug(f"=====> id: {new_conv.id} ; g={grouping}, w_after={kernel.shape}, ic={self.in_channels}")
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
        out_channel_sizes=None,
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
            out_channel_sizes=out_channel_sizes,
        )
        self.bn = ElasticWidthBatchnorm1d(out_channels, track_running_stats)
        self.norm = True
        self.act = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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

        padding = conv1d_get_padding(kernel_size, dilation)

        # self.in_channels = kernel.size(1)
        # self.out_channels = kernel.size(0)
        self.set_in_and_out_channel(kernel)

        new_conv = ConvBn1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=dilation,
            bias=False,
            groups=grouping
        )
        tmp_bn = self.bn.get_basic_batchnorm1d()

        new_conv.last_grouping_param = self.groups
        if not hasattr(new_conv, 'id'):
            new_conv.id = "ElasticConvBn1d-" + str(random.randint(0, 1000)*2000)
            logging.debug(f"Validation id created: {new_conv.id} ; g={grouping}, w_before={kernel.shape}, ic={self.in_channels}")
        else:
            logging.debug("id already present: {new_conv.id}")
        kernel, _ = adjust_weight_if_needed(module=new_conv, kernel=kernel, groups=new_conv.groups, in_place_adjustment=False)
        logging.debug(f"=====> id: {new_conv.id} ; g={grouping}, w_after={kernel.shape}, ic={self.in_channels}")

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
        out_channel_sizes=None,
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
            out_channel_sizes=out_channel_sizes,
        )

        self.relu = ElasticPermissiveReLU()
        self.norm = True
        self.act = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.relu(super(ElasticConvBnReLu1d, self).forward(input))

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        dilation = self.get_dilation_size()
        grouping = self.get_group_size()

        #self.in_channels = kernel.size(1)
        #self.out_channels = kernel.size(0)
        self.set_in_and_out_channel(kernel)

        padding = conv1d_get_padding(kernel_size, dilation)
        new_conv = ConvBnReLu1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=dilation,
            bias=False,
            groups=grouping
        )
        tmp_bn = self.bn.get_basic_batchnorm1d()

        new_conv.last_grouping_param = self.groups
        if not hasattr(new_conv, 'id'):
            new_conv.id = "ElasticConvBnReLu1d-" + str(random.randint(0, 1000)*2000)
            logging.debug(f"Validation id created: {new_conv.id} ; g={grouping}, w_before={kernel.shape}, ic={self.in_channels}")
        else:
            logging.debug("id already present: {new_conv.id}")
        kernel, _ = adjust_weight_if_needed(module=new_conv, kernel=kernel, groups=new_conv.groups, in_place_adjustment=False)
        logging.info(f"=====> id: {new_conv.id} ; g={grouping}, w_after={kernel.shape}, ic={self.in_channels}")
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
            groups=groups,
            bias=bias,
        )
        self.relu = nn.ReLU()
        self.norm = False
        self.act = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if(self.groups == 1):
            return self.relu(super(ConvRelu1d, self).forward(input))

        logging.debug(f"Groups in forward: {self.groups}")
        full_kernel = torch.ones(self.weight.shape, device=self.weight.device)
        full_kernel.copy_(self.weight)
        # if(self.groups > 1):
        #     self.weight = nn.Parameter(adjust_weights_for_grouping(self.weight, self.groups))
        # TODO MR ist das so nötig ?
        #adjust_weight_if_needed(module=self, kernel=self.weight, groups=self.groups, in_place_adjustment=True)

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
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels, track_running_stats=track_running_stats)
        self.norm = True
        self.act = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if(self.groups == 1):
            return self.bn(super(ConvBn1d, self).forward(input))

        logging.debug(f"Groups in forward: {self.groups}")
        full_kernel = torch.ones(self.weight.shape, device=self.weight.device)
        full_kernel.copy_(self.weight)
        # if(self.groups > 1):
        #     self.weight = nn.Parameter(adjust_weights_for_grouping(self.weight, self.groups))
        # TODO MR ist das so nötig ?
        #adjust_weight_if_needed(module=self, kernel=self.weight, groups=self.groups, in_place_adjustment=True)

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
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels, track_running_stats=track_running_stats)
        self.relu = nn.ReLU()
        self.norm = True
        self.act = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if(self.groups == 1):
            return self.relu(super(ConvBnReLu1d, self).forward(input))

        logging.debug(f"Groups in forward: {self.groups}")
        # TODO MR ist das so nötig ?
        full_kernel = torch.ones(self.weight.shape, device=self.weight.device)
        full_kernel.copy_(self.weight)
        # if(self.groups > 1):
        #     self.weight = nn.Parameter(adjust_weights_for_grouping(self.weight, self.groups))
        #adjust_weight_if_needed(module=self, kernel=self.weight, groups=self.groups, in_place_adjustment=True)

        tensor = self.relu(super(ConvBnReLu1d, self).forward(input))
        self.weight = nn.Parameter(full_kernel)
        return tensor
