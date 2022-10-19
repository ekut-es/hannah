import copy
import logging
from pyclbr import Function
from typing import List

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as nnf

from hannah.models.factory import qat

from ..utilities import (
    conv1d_get_padding,
    filter_primary_module_weights,
    filter_single_dimensional_weights,
    get_kernel_for_dsc,
    prepare_kernel_for_depthwise_separable_convolution,
    prepare_kernel_for_pointwise_convolution,
    sub_filter_start_end,
    getGroups,
    adjust_weights_for_grouping
)


# It's a wrapper for a convolutional layer that allows for the number of input and
# output channels to be changed
class _Elastic:

    def __init__(self, in_channel_filter, out_channel_filter, out_channel_sizes=None):
        self.in_channel_filter: int = in_channel_filter
        self.out_channel_filter: int = out_channel_filter
        self.out_channel_sizes: List[int] = out_channel_sizes

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Module:
        return None

    def get_out_channel_sizes(self):
        return self.out_channel_sizes

    # return a safe copy of a conv1d equivalent to this module in the current state
    def assemble_basic_module(self) -> nn.Module:  # Module, so that Sequentials are possible, like DSC
        return copy.deepcopy(self.get_basic_module())

    def set_out_channel_filter(self, out_channel_filter):
        if out_channel_filter is not None:
            self.out_channel_filter = out_channel_filter
            if hasattr(self, "bn") and hasattr(self.bn, "__iter__"):
                for element in self.bn:
                    element.channel_filter = out_channel_filter
            elif hasattr(self, "bn") and not hasattr(self.bn, "__iter__"):
                self.bn.channel_filter = out_channel_filter

    def set_in_channel_filter(self, in_channel_filter):
        if in_channel_filter is not None:
            self.in_channel_filter = in_channel_filter


# It's a 1D convolutional layer that can change its kernel size and dilation size
class ElasticBase1d(nn.Conv1d, _Elastic):

    res_break : bool = False

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
        padding_mode: str = "zeros",
        out_channel_sizes=None,
    ):
        # sort available kernel sizes from largest to smallest (descending order)
        kernel_sizes.sort(reverse=True)
        # make sure 0 is not set as kernel size. Must be at least 1
        if 0 in kernel_sizes:
            kernel_sizes.remove(0)
        self.kernel_sizes: List[int] = kernel_sizes
        # after sorting kernel sizes, the maximum and minimum size available are the first and last element
        self.max_kernel_size: int = kernel_sizes[0]
        self.min_kernel_size: int = kernel_sizes[-1]
        # initially, the target size is the full kernel
        self.target_kernel_index: int = 0

        # sort available dilation sizes from largest to smallest (descending order)
        dilation_sizes.sort(reverse=False)
        # make sure 0 is not set as dilation size. Must be at least 1
        if 0 in dilation_sizes:
            dilation_sizes.remove(0)
        self.dilation_sizes: List[int] = dilation_sizes
        # after sorting dilation sizes, the maximum and minimum size available are the first and last element
        self.max_dilation_size: int = dilation_sizes[-1]
        self.min_dilation_size: int = dilation_sizes[0]
        # initially, the target size is the smallest dilation (1)
        self.target_dilation_index: int = 0

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels

        # dynamic width changes the in and out_channels
        # hence we save then here
        self.initial_in_channels : int = in_channels
        self.initial_out_channels : int = out_channels

        # sort available grouping sizes from largest to smallest (descending order)
        groups.sort(reverse=False)
        # make sure 0 is not set as grouping size. Must be at least 1
        if 0 in groups:
            groups.remove(0)

        self.group_sizes: List[int] = groups

        self.max_group_size: int = self.group_sizes[-1]
        self.min_group_size: int = self.group_sizes[0]
        self.target_group_index: int = 0
        self.last_grouping_param = self.get_group_size()

        # set the groups value in the model
        self.groups = self.get_group_size()

        # sort available grouping sizes from largest to smallest (descending order)
        dscs.sort(reverse=False)
        # make sure 0 is not set as grouping size. Must be at least 1
        self.dscs: List[bool] = dscs

        self.max_dsc: bool = self.dscs[-1]
        self.min_dsc: bool = self.dscs[0]
        self.target_dsc_index: int = 0

        # store first grouping param -
        # needed for speedup of check if weight needs to be adjusted for grouping
        self.last_dsc_param = self.get_dsc()

        # set the groups value in the model
        self.dsc_on = self.get_dsc()

        self.padding = conv1d_get_padding(
            self.kernel_sizes[self.target_kernel_index],
            self.dilation_sizes[self.target_dilation_index],
        )
        nn.Conv1d.__init__(
            self,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.max_kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation_sizes[self.target_dilation_index],
            groups=self.group_sizes[self.target_group_index],
            bias=bias,
        )

        _Elastic.__init__(
            self, [True] * in_channels, [True] * out_channels, out_channel_sizes
        )

        # the list of kernel transforms will have one element less than the list of kernel sizes.
        # between every two sequential kernel sizes, there will be a kernel transform
        # the subsequent kernel is determined by applying the same-size center of the previous kernel to the transform
        self.kernel_transforms = nn.ModuleList([])
        for i in range(len(kernel_sizes) - 1):
            # the target size of the kernel transform is the next kernel size in the sequence
            new_kernel_size = kernel_sizes[i + 1]
            # kernel transform is kept minimal by being shared between channels.
            # It is simply a linear transformation from the center of the previous kernel to the new kernel
            # directly applying the kernel to the transform is possible: nn.Linear accepts
            # multi-dimensional input in a way where the last input dim is transformed
            # from in_channels to out_channels for the last output dim
            new_transform_module = nn.Linear(
                new_kernel_size, new_kernel_size, bias=False
            )
            # initialise the transform as the identity matrix to start training
            # from the center of the larger kernel
            new_transform_module.weight.data.copy_(torch.eye(new_kernel_size))
            # transform weights are initially frozen
            new_transform_module.weight.requires_grad = True
            self.kernel_transforms.append(new_transform_module)
        self.set_kernel_size(self.max_kernel_size)
        """
        self.dilation_transforms = nn.ModuleList()
        for k in range(len(kernel_sizes)):
            self.dilation_transforms.append(nn.ModuleList())
            for i in range(len(dilation_sizes) - 1):
                new_transform_module = nn.Linear(
                    self.kernel_sizes[k], self.kernel_sizes[k], bias=False
                )
                # initialise the transform as the identity matrix to start training
                # from the center of the larger kernel
                new_transform_module.weight.data.copy_(torch.eye(self.kernel_sizes[k]))
                # transform weights are initially frozen
                new_transform_module.weight.requires_grad = True
                self.dilation_transforms[k].append(new_transform_module)
        """
        self.update_padding()

    def set_bn_parameter(self, conv : nn.Conv1d, tmp_bn, num_tracked):
        """
        Caller for BatchNorm Parameters
        This unifies the call in the different methods, especially in dsc / not dsc forward
        And assigns the attributes in tmp_bn to the param conv
        """
        conv.bn.num_features = tmp_bn.num_features
        conv.bn.weight = tmp_bn.weight
        conv.bn.bias = tmp_bn.bias
        conv.bn.running_var = tmp_bn.running_var
        conv.bn.running_mean = tmp_bn.running_mean
        conv.bn.num_batches_tracked = num_tracked
        return conv

    def prepare_dsc_for_validation_model(
        self,
        conv_class : nn.Module,
        full_kernel,
        full_bias,
        in_channels,
        out_channels,
        grouping,
        stride,
        padding,
        dilation,
        # for quant
        qconfig=None,
        out_quant=None,
        bn_eps=None,
        bn_momentum=None,
        bn_caller : tuple = None,
    ):
        """
        This method creates the necessary validation models for DSC.
        It creates the validation model as torch.Sequence of standard pytorch convolution models.
        The structure is analog to the DSC method do_dsc.
        This method can also handle quantization models.

        :param: conv_class: the conv class that should be used for DSC
        :param: bn_eps: batchnorm parameter
        :param: bn_momentum:  batchnorm parameter
        :param: bn_caller: batchnorm_caller, for applying batch norm to the conv_class
        """

        # use qconfig and out_quant parameters
        is_quant = qconfig is not None and out_quant is not None
        uses_batch_norm = bn_eps is not None and bn_momentum is not None

        # depthwise
        filtered_kernel_depth, bias = prepare_kernel_for_depthwise_separable_convolution(
             self,
             kernel=full_kernel,
             bias=full_bias,
             in_channels=in_channels
        )
        in_channel_depth = in_channels

        param_depthwise_conv : dict = {
            "in_channels": in_channel_depth,
            "out_channels": filtered_kernel_depth.size(0),
            "kernel_size": filtered_kernel_depth.size(2),
            "bias": bias,
            "groups": in_channel_depth,
            "padding": padding
        }

        # is either quant Conv1d or normal Conv1d
        depthwise_separable = nn.Conv1d(
                **param_depthwise_conv
        )
        depthwise_separable.weight.data = filtered_kernel_depth

        # pointwise  convolution
        kernel, bias = self.get_kernel()
        filtered_kernel_point = prepare_kernel_for_pointwise_convolution(
            kernel=kernel,
            grouping=grouping
        )

        param_point_conv : dict = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": filtered_kernel_point.size(2),
            "bias": bias,
            "groups": grouping,
            "stride": stride,
            "dilation": dilation,
        }
        if is_quant:
            param_point_conv["qconfig"] = qconfig
            param_point_conv["out_quant"] = out_quant

        if is_quant and uses_batch_norm:
            param_point_conv["eps"] = bn_eps
            param_point_conv["momentum"] = bn_momentum

        # create a convolution model, from the kwargs params above
        pointwise = conv_class(
                **param_point_conv
        )
        pointwise.weight.data = filtered_kernel_point

        if bn_caller:
            # if batchnorm should be applied call bn_caller
            bn_function, bn_norm, num_tracked = bn_caller
            pointwise = bn_function(pointwise, bn_norm, num_tracked)

        depthwise_separable_conv = nn.Sequential(depthwise_separable, pointwise)

        return depthwise_separable_conv

    def do_dsc(
        self,
        input,
        full_kernel,
        full_bias,
        grouping,
        stride,
        padding,
        dilation,
        quant_weight=None,
        quant_bias=None,
        quant_weight_function : Function = None,  # for quantization
        quant_bias_function : Function = None,  # for quantization
    ):

        """
            This  method will perform the DSC(=Depthwise Separable Convolution).
            This  method can also handle quantized models.
            DSC is done in two steps:
            1. Depthwise Separable: Set Group = In_Channels, Output = k*In_Channels
            2. Pointwise Convolution, with Grouping = Grouping-Param und Out_Channel = Out_Channel-Param

            The Params above are used for quantized models
            :param: quant_weight: quantized weights
            :param: quant_bias: quantized zero_bias
            :param: quant_weight_function: fake_quant function for weight
            :param: quant_bias_function: fake_quant function for bias
        """
        use_fake_weight = quant_weight_function is not None
        use_fake_bias = quant_bias_function is not None and full_bias is not None

        # get the **actual** count of in_channels
        in_channels = self.in_channel_filter.count(True)

        filtered_kernel, bias = prepare_kernel_for_depthwise_separable_convolution(
             self,
             kernel=full_kernel,
             bias=full_bias,
             in_channels=in_channels
        )
        # params for depthwise separable convolution
        param_depthwise_conv : dict = {
            "input": input,
            "weight": filtered_kernel if use_fake_weight is False else quant_weight_function(filtered_kernel),
            "bias": bias if use_fake_bias is False else quant_bias_function(bias),
            "groups": in_channels,
            "padding": padding
        }
        # do depthwise
        res_depthwise = nnf.conv1d(
            **param_depthwise_conv
            # Important!! Kein Stride, keine Dilation, da sonst der Effekt von Depthwise daneben geht.
            # Dies kann dann im nächsten Step nachgeholt werden. Sonst stimmt der Output nicht.
        )
        use_quant = torch.is_tensor(quant_weight) and torch.is_tensor(quant_bias)

        if use_quant:
            kernel, bias = quant_weight, quant_bias
            filtered_kernel = get_kernel_for_dsc(kernel)
        else:
            kernel, bias = self.get_kernel()
            filtered_kernel = prepare_kernel_for_pointwise_convolution(
                kernel=kernel,
                grouping=grouping
            )

        # pointwise convolution
        param_point_conv : dict = {
            "input": res_depthwise,
            "weight": filtered_kernel if use_fake_weight is False else quant_weight_function(filtered_kernel),
            "bias": bias if use_fake_bias is False else quant_bias_function(bias),
            "groups": grouping,
            "stride": stride,
            "dilation": dilation,
        }

        res_pointwise = nnf.conv1d(
            **param_point_conv
        )
        return res_pointwise

    def set_in_and_out_channel(self, kernel, filtered : bool = True):
        """
        This method uses the kernel for setting the input and outputchannel
        if dynamic width is activated (channelfilters), the amount of channels is reduced,
        hence we can't use the initial values (self.(in/out)_channel) of the constructor

        This method sets the self.(in/out)_channel value to the right amount of channels
        extracted from the kernel that will be used.

        if filtered is False, the self.initial_(in/out)_channels will be used.

        The previous values will be stored in the attribute prev_in_channels and prev_out_channels.

        :param: kernel : the weights , size(1) for in channel, size(0) for out_channels
        :param: filtered: use kernel size data for setting the (in/out) channels
        """
        self.prev_in_channels = self.in_channels
        self.prev_out_channels = self.out_channels
        self.in_channels = kernel.size(1) if filtered else self.initial_in_channels
        self.out_channels = kernel.size(0) if filtered else self.initial_out_channels

    def reset_in_and_out_channel_to_previous(self):
        """
        Analog to set_in_and_out_channels:
        Resets the in and out_channels
        """
        self.in_channels = self.prev_in_channels
        self.out_channels = self.prev_out_channels

    def set_kernel_size(self, new_kernel_size):
        """
        If the requested kernel size is outside of the min/max range, clamp it to
        the min/max range. If the requested kernel size is not an available kernel
        size, default to the max kernel size

        :param new_kernel_size: the size of the kernel you want to use
        """
        # previous_kernel_size = self.kernel_sizes[self.target_kernel_index]
        if (
            new_kernel_size < self.min_kernel_size
            or new_kernel_size > self.max_kernel_size
        ):
            logging.warn(
                f"requested elastic kernel size ({new_kernel_size}) outside of min/max range: ({self.max_kernel_size}, {self.min_kernel_size}). clamping."
            )
            if new_kernel_size <= self.min_kernel_size:
                new_kernel_size = self.min_kernel_size
            else:
                new_kernel_size = self.max_kernel_size

        self.target_kernel_index = 0
        try:
            index = self.kernel_sizes.index(new_kernel_size)
            self.target_kernel_index = index
            self.kernel_size = new_kernel_size
            self.update_padding()

        except ValueError:
            logging.warn(
                f"requested elastic kernel size {new_kernel_size} is not an available kernel size. Defaulting to full size ({self.max_kernel_size})"
            )

        # if self.kernel_sizes[self.target_kernel_index] != previous_kernel_size:
        # print(f"\nkernel size was changed: {previous_kernel_size} -> {self.kernel_sizes[self.target_kernel_index]}")

    # the initial kernel size is the first element of the list of available sizes
    # set the kernel back to its initial size
    def reset_kernel_size(self):
        self.set_kernel_size(self.kernel_sizes[0])

    # step current kernel size down by one index, if possible.
    # return True if the size limit was not reached
    def step_down_kernel_size(self):
        next_kernel_index = self.target_kernel_index + 1
        if next_kernel_index < len(self.kernel_sizes):
            self.set_kernel_size(self.kernel_sizes[next_kernel_index])
            # print(f"stepped down kernel size of a module! Index is now {self.target_kernel_index}")
            return True
        else:
            logging.debug(
                f"unable to step down kernel size, no available index after current: {self.target_kernel_index} with size: {self.kernel_sizes[self.target_kernel_index]}"
            )
            return False

    def pick_kernel_index(self, target_kernel_index: int):
        if (target_kernel_index < 0) or (
            target_kernel_index >= len(self.kernel_sizes)):
            logging.warn(
                f"selected kernel index {target_kernel_index} is out of range: 0 .. {len(self.kernel_sizes)}. Setting to last index."
            )
            target_kernel_index = len(self.kernel_sizes) - 1
        self.set_kernel_size(self.kernel_sizes[target_kernel_index])

    def get_available_kernel_steps(self):
        return len(self.kernel_sizes)

    def get_full_width_kernel(self):
        """
        It applies the kernel transformations to the kernel until the target kernel
        index is reached
        :return: The found target kernel.
        """
        current_kernel_index = 0
        current_dilation_index = 1
        current_kernel = self.weight

        logging.debug("Target kernel index: %s", str(self.target_kernel_index))

        # step through kernels until the target index is reached.
        while current_kernel_index < self.target_kernel_index:
            if current_kernel_index >= len(self.kernel_sizes):
                logging.warn(
                    f"kernel size index {current_kernel_index} is out of range. Elastic kernel acquisition stopping at last available kernel"
                )
                break
            # find start, end pos of the kernel center for the given next kernel size
            start, end = sub_filter_start_end(
                self.kernel_sizes[current_kernel_index],
                self.kernel_sizes[current_kernel_index + 1],
            )
            # extract the kernel center of the correct size
            kernel_center = current_kernel[:, :, start:end]
            # apply the kernel transformation to the next kernel. the n-th transformation
            # is applied to the n-th kernel, yielding the (n+1)-th kernel
            next_kernel = self.kernel_transforms[current_kernel_index](
                kernel_center)
            # the kernel has now advanced through the available sizes by one
            current_kernel = next_kernel
            current_kernel_index += 1

        # step through dilation until the target index is reached.
        """
        while current_dilation_index < self.target_dilation_index:
            if current_dilation_index >= len(self.dilation_sizes):
                logging.warn(
                    f"kernel size index {current_kernel_index} is out of range. Elastic kernel acquisition stopping at last available kernel"
                )
                break
            # apply the kernel transformation to the next kernel. the n-th transformation
            # is applied to the n-th kernel, yielding the (n+1)-th kernel
            next_kernel = self.dilation_transforms[self.target_kernel_index][
                current_dilation_index
            ](current_kernel)
            # the kernel has now advanced through the available sizes by one
            current_kernel = next_kernel
            current_dilation_index += 1
        """

        return current_kernel

    def get_kernel(self):
        """
        If the input and output channels are not filtered, the full kernel is
        returned. Otherwise, the kernel is filtered using the input and output
        channel filters. If the module has a bias parameter, it is also filtered
        using the output channel filter
        :return: The new kernel and bias.
        """
        full_kernel = self.get_full_width_kernel()
        new_kernel = None
        if all(self.in_channel_filter) and all(self.out_channel_filter):
            # if no channel filtering is required, the full kernel can be kept
            new_kernel = full_kernel
        else:
            # if channels need to be filtered, apply filters to the kernel
            new_kernel = filter_primary_module_weights(
                full_kernel, self.in_channel_filter, self.out_channel_filter
            )
        # if the module has a bias parameter, also apply the output filtering to it.
        if self.bias is None:
            return new_kernel, None
        else:
            new_bias = filter_single_dimensional_weights(
                self.bias, self.out_channel_filter
            )
            return new_kernel, new_bias

    def set_dilation_size(self, new_dilation_size):
        if (
            new_dilation_size < self.min_dilation_size
            or new_dilation_size > self.max_dilation_size
        ):
            logging.warn(
                f"requested elastic dilation size ({new_dilation_size}) outside of min/max range: ({self.max_dilation_size}, {self.min_dilation_size}). clamping."
            )
            if new_dilation_size < self.min_dilation_size:
                new_dilation_size = self.min_dilation_size
            else:
                new_dilation_size = self.max_dilation_size

        self.target_dilation_index = 0
        try:
            index = self.dilation_sizes.index(new_dilation_size)
            self.target_dilation_index = index
            self.dilation = self.dilation_sizes[self.target_dilation_index]
            self.update_padding()

        except ValueError:
            logging.warn(
                f"requested elastic dilation size {new_dilation_size} is not an available dilation size. Defaulting to full size ({self.max_dilation_size})"
            )

    def update_padding(self):
        self.padding = conv1d_get_padding(self.kernel_size, self.dilation)

    # the initial dilation size is the first element of the list of available sizes
    # set the dilation back to its initial size
    def reset_dilation_size(self):
        self.set_dilation_size(self.dilation_sizes[0])

    # step current kernel size down by one index, if possible.
    # return True if the size limit was not reached
    def step_down_dilation_size(self):
        next_dilation_index = self.target_dilation_index + 1
        if next_dilation_index < len(self.dilation_sizes):
            self.set_dilation_size(self.dilation_sizes[next_dilation_index])
            return True
        else:
            logging.debug(
                f"unable to step down dilation size, no available index after current: {self.target_dilation_index} with size: {self.dilation_sizes[self.target_dilation_index]}"
            )
            return False

    def pick_dilation_index(self, target_dilation_index: int):
        if (target_dilation_index < 0) or (
            target_dilation_index >= len(self.dilation_sizes)
        ):
            # MR-Optional Change (can be done in master)
            logging.warn(
                f"selected dilation index {target_dilation_index} is out of range: 0 .. {len(self.dilation_sizes)}. Setting to last index."
            )
            target_dilation_index = len(self.dilation_sizes) - 1
        self.set_dilation_size(self.dilation_sizes[target_dilation_index])

    def get_available_dilation_steps(self):
        return len(self.dilation_sizes)

    def get_available_grouping_steps(self):
        return len(self.group_sizes)

    def get_available_dsc_steps(self):
        return len(self.dscs)

    def get_dilation_size(self):
        return self.dilation_sizes[self.target_dilation_index]

    def pick_group_index(self, target_group_index: int):
        if (target_group_index < 0) or (
            target_group_index >= len(self.group_sizes)
        ):
            logging.warn(
                f"selected group index {target_group_index} is out of range: 0 .. {len(self.group_sizes)}. Setting to last index."
            )
            target_group_index = len(self.group_sizes) - 1
        self.set_group_size(self.group_sizes[target_group_index])

    def pick_dsc_index(self, target_dsc_index: int):
        if (target_dsc_index < 0) or (
            target_dsc_index >= len(self.dscs)
        ):
            logging.warn(
                f"selected dsc index {target_dsc_index} is out of range: 0 .. {len(self.dscs)}. Setting to last index."
            )
            target_dsc_index = len(self.dscs) - 1
        self.set_dsc(self.dscs[target_dsc_index])

    # the initial group size is the first element of the list of available sizes
    # resets the group size back to its initial size
    def reset_group_size(self):
        self.set_group_size(self.group_sizes[0])

    def reset_dscs(self):
        self.set_dsc(self.dscs[0])

    def get_group_size(self):
        return self.group_sizes[self.target_group_index]

    def get_dsc(self):
        return self.dscs[self.target_dsc_index]

    def set_group_size(self, new_group_size):
        if (
            new_group_size < self.min_group_size
            or new_group_size > self.max_group_size
        ):
            logging.warn(
                f"requested elastic group size ({new_group_size}) outside of min/max range: ({self.max_group_size}, {self.min_group_size}). clamping."
            )
            if new_group_size < self.min_group_size:
                new_group_size = self.min_group_size
            else:
                new_group_size = self.max_group_size

        self.target_group_index = 0
        try:
            index = self.group_sizes.index(new_group_size)
            self.target_group_index = index
            # if hasattr(self, 'from_skipping') and self.from_skipping is True:
            #     logging.warn(f"setting groupsizes from skipping is: {self.from_skipping}")
            # else:
            #     self.groups = self.group_sizes[index]

        except ValueError:
            logging.warn(
                f"requested elastic group size {new_group_size} is not an available group size. Defaulting to full size ({self.max_group_size})"
            )

    def set_dsc(self, new_dsc):
        if (
            new_dsc < self.min_dsc
            or new_dsc > self.max_dsc
        ):
            logging.warn(
                f"requested elastic dsc ({new_dsc}) outside of min/max range: ({self.max_dsc}, {self.min_dsc}). clamping."
            )
            if new_dsc < self.min_dsc:
                new_dsc = self.min_dsc
            else:
                new_dsc = self.max_dsc

        self.target_dsc_index = 0
        try:
            index = self.dscs.index(new_dsc)
            self.target_dsc_index = index
            # if hasattr(self, 'from_skipping') and self.from_skipping is True:
            #     logging.warn(f"setting groupsizes from skipping is: {self.from_skipping}")
            # else:
            #     self.groups = self.group_sizes[index]

        except ValueError:
            logging.warn(
                f"requested elastic dsc {new_dsc} is not an available group size. Defaulting to full size ({self.max_dsc})"
            )

    # step current kernel size down by one index, if possible.
    # return True if the size limit was not reached
    def step_down_group_size(self):
        next_group_index = self.target_group_index + 1
        if next_group_index < len(self.group_sizes):
            self.set_group_size(self.group_sizes[next_group_index])
            # print(f"stepped down group size of a module! Index is now {self.target_group_index}")
            return True
        else:
            logging.debug(
                f"unable to step down group size, no available index after current: {self.target_group_index} with size: {self.group_sizes[self.target_group_index]}"
            )
            return False

    def step_down_dsc(self):
        next_dsc_index = self.target_dsc_index + 1
        if next_dsc_index < len(self.dscs):
            self.set_dsc(self.dscs[next_dsc_index])
            # print(f"stepped down group size of a module! Index is now {self.target_group_index}")
            return True
        else:
            logging.debug(
                f"unable to step down dsc, no available index after current: {self.target_dsc_index} with size: {self.dscs[self.target_dsc_index]}"
            )
            return False

    # TODO:
    # MR: not in use at the moment - but maybe later
    # def getGrouping(self):
    #     """"
    #          Returns a possible Grouping using GCD
    #      """
    #     gcd_input_output = np.gcd(self.in_channels, self.out_channels)
    #     self.group_sizes = getGroups(gcd_input_output)
    #     logging.info(
    #         f"InputSize: {self.in_channels},  OutputSize: {self.out_channels}")
    #     logging.info(f"GCD: {gcd_input_output}")
    #     for group in self.group_sizes:
    #         grouping_possible_input = self.in_channels % group == 0
    #         grouping_possible_output = self.out_channels % group == 0
    #         logging.info(
    #             f"Grouping: {group}, Possible Grouping(I,O)? ({grouping_possible_input},{grouping_possible_output})")
    #         if not (grouping_possible_output and grouping_possible_input):
    #             self.group_sizes.remove(group)
    #     return self.group_sizes

    # Wrapper Class
    def adjust_weights_for_grouping(self, weights, input_divided_by):
        return adjust_weights_for_grouping(weights, input_divided_by)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

    def extra_repr(self):
        # TODO(jerryzh): extend
        pass
        # return super(ElasticBase1d, self).extra_repr()
