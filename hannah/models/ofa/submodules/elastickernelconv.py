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
)
from .elasticBatchnorm import ElasticWidthBatchnorm1d
from .elasticLinear import ElasticPermissiveReLU



# TODO Validation
class ElasticConv1d(ElasticBase1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        dilation_sizes: List[int],
        groups: List[int],
        dscs: List[bool],
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
            dscs=dscs,
            bias=bias,
            out_channel_sizes=out_channel_sizes,
        )
        self.norm = False
        self.act = False
        # TODO es wäre auch möglich das ganze als Flag einzubauen wie norm und act, aber hier wäre die Frage wie man es mit dem trainieren macht ?
        # So wäre es statisch und nicht wirklich sinnvoll

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # get the kernel for the current index
        kernel, bias = self.get_kernel()
        # First get the correct count of in and outchannels
        self.set_in_and_out_channel(kernel)

        dilation = self.get_dilation_size()
        # get padding for the size of the kernel
        padding = conv1d_get_padding(
            self.kernel_sizes[self.target_kernel_index], dilation
        )
        grouping = self.get_group_size()
        # Hier muss dann wenn DSC on ist, die Logik implementiert werden dass DSC komplett greift
        dsc = self.get_dsc()

        if dsc is False:
            kernel, _ = adjust_weight_if_needed(module=self, kernel=kernel, groups=grouping)
            return nnf.conv1d(input, kernel, bias, self.stride, padding, dilation, grouping)
        else:
            # we use the full kernel here, because if the input_channel_size is greater than the output_channel_size
            # we have to increase the output_channel_size for dpc, hence we need the full kernel, because, the filtered kernel
            # is in that particular case to small.
            kernel, bias = self.get_full_width_kernel(), self.bias   # if self.in_channels > self.out_channels else (kernel, bias)
            # kernel, bias = self.get_full_width_kernel(), self.bias   if self.in_channels > self.out_channels else (kernel, bias)
            return self.do_dpc(input, kernel=kernel, bias=bias, grouping=grouping, stride=self.stride, padding=padding, dilation=dilation)

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Conv1d:
        # TODO MR Validaiton model needs to be done after normal thing works

        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        dilation = self.get_dilation_size()
        ##
        grouping = self.get_group_size()

        padding = conv1d_get_padding(kernel_size, dilation)

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

        # for ana purposes handy - set a unique id so we can track this specific convolution
        if not hasattr(new_conv, 'id'):
            new_conv.id = "ElasticConv1d-" + str(random.randint(0, 1000)*2000)
            logging.debug(f"Validation id created: {new_conv.id} ; g={grouping}, w_before={kernel.shape}, ic={self.in_channels}")
        else:
            logging.debug("Validation id already present: {new_conv.id}")

        kernel, _ = adjust_weight_if_needed(module=new_conv, kernel=kernel, groups=new_conv.groups)
        new_conv.weight.data = kernel
        if bias is not None:
            new_conv.bias = bias

        logging.debug(f"=====> id: {new_conv.id} ; g={grouping}, w_after={kernel.shape}, ic={self.in_channels}")
        return new_conv


class ElasticConvReLu1d(ElasticBase1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        dilation_sizes: List[int],
        groups: List[int],
        dscs: List[bool],
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
            dscs=dscs,
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
        # First get the correct count of in and outchannels
        # given by the kernel (after setting the kernel correctly, with the help of input-/output_filters)
        self.set_in_and_out_channel(kernel)
        dilation = self.get_dilation_size()
        # get padding for the size of the kernel
        padding = conv1d_get_padding(
            self.kernel_sizes[self.target_kernel_index], dilation
        )

        self.get_c
        grouping = self.get_group_size()
        # Hier muss dann wenn DSC on ist, die Logik implementiert werden dass DSC komplett greift
        dsc_on = self.get_dsc()
        if dsc_on is False:
            kernel, _ = adjust_weight_if_needed(module=self, kernel=kernel, groups=grouping)
            output = nnf.conv1d(input, kernel, bias, self.stride, padding, dilation, grouping)
        else:
            output = self.do_dpc(input, kernel=kernel, bias=bias, grouping=grouping, stride=self.stride, padding=padding, dilation=dilation)

        return self.relu(
            output
        )

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        dilation = self.get_dilation_size()
        grouping = self.get_group_size()

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

        # for ana purposes handy - set a unique id so we can track this specific convolution
        new_conv.last_grouping_param = self.groups
        if not hasattr(new_conv, 'id'):
            new_conv.id = "ConvRelu1d-" + str(random.randint(0, 1000)*2000)
            logging.debug(f"Validation id created: {new_conv.id} ; g={grouping}, w_before={kernel.shape}, ic={self.in_channels}")
        else:
            logging.debug("Validation id already present: {new_conv.id}")

        kernel, _ = adjust_weight_if_needed(module=new_conv, kernel=kernel, groups=new_conv.groups)
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
        dscs: List[bool],
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
            dscs=dscs,
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

        # for ana purposes handy - set a unique id so we can track this specific convolution
        new_conv.last_grouping_param = self.groups
        if not hasattr(new_conv, 'id'):
            new_conv.id = "ElasticConvBn1d-" + str(random.randint(0, 1000)*2000)
            logging.debug(f"Validation id created: {new_conv.id} ; g={grouping}, w_before={kernel.shape}, ic={self.in_channels}")
        else:
            logging.debug("id already present: {new_conv.id}")
        kernel, _ = adjust_weight_if_needed(module=new_conv, kernel=kernel, groups=new_conv.groups)
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
        dscs: List[bool],
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        track_running_stats=False,
        out_channel_sizes=None,
        from_skipping=False,
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
            dscs=dscs,
            bias=bias,
            out_channel_sizes=out_channel_sizes,
        )

        self.relu = ElasticPermissiveReLU()
        self.norm = True
        self.act = True
        self.from_skipping = from_skipping

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.relu(super(ElasticConvBnReLu1d, self).forward(input))

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        dilation = self.get_dilation_size()
        grouping = self.get_group_size()

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

        # for ana purposes handy - set a unique id so we can track this specific convolution
        new_conv.last_grouping_param = self.groups
        if not hasattr(new_conv, 'id'):
            new_conv.id = "ElasticConvBnReLu1d-" + str(random.randint(0, 1000)*2000)
            logging.debug(f"Validation id created: {new_conv.id} ; g={grouping}, w_before={kernel.shape}, ic={self.in_channels}")
        else:
            logging.debug("id already present: {new_conv.id}")
        kernel, _ = adjust_weight_if_needed(module=new_conv, kernel=kernel, groups=new_conv.groups)
        logging.debug(f"=====> id: {new_conv.id} ; g={grouping}, w_after={kernel.shape}, ic={self.in_channels}, fromSkipping={self.from_skipping}")

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
        dsc: bool = False,
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
        return self.relu(super(ConvRelu1d, self).forward(input))


class ConvBn1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dsc: bool = False,
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
        return self.bn(super(ConvBn1d, self).forward(input))


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
        dsc: bool = False,
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
        return self.relu(super(ConvBnReLu1d, self).forward(input))

