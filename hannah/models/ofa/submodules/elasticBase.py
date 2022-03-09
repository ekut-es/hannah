import copy
from typing import List
import torch.nn as nn
import logging
import torch
from ..utilities import (
    conv1d_get_padding,
    filter_primary_module_weights,
    filter_single_dimensional_weights,
    sub_filter_start_end,
)


class _Elastic:
    def __init__(self, in_channel_filter, out_channel_filter):
        self.in_channel_filter: int = in_channel_filter
        self.out_channel_filter: int = out_channel_filter

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Conv1d:
        return None

    # return a safe copy of a conv1d equivalent to this module in the current state
    def assemble_basic_module(self) -> nn.Conv1d:
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


class ElasticBase1d(nn.Conv1d, _Elastic):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        dilation_sizes: List[int],
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
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
            groups=groups,
            bias=bias,
        )

        _Elastic.__init__(self, [True] * in_channels, [True] * out_channels)

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

    def set_kernel_size(self, new_kernel_size):
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
        if (target_kernel_index < 0) or (target_kernel_index >= len(self.kernel_sizes)):
            logging.warn(
                f"selected kernel index {target_kernel_index} is out of range: 0 .. {len(self.kernel_sizes)}. Setting to last index."
            )
            target_kernel_index = len(self.kernel_sizes) - 1
        self.set_kernel_size(self.kernel_sizes[target_kernel_index])

    def get_available_kernel_steps(self):
        return len(self.kernel_sizes)

    def get_full_width_kernel(self):
        current_kernel_index = 0
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
            next_kernel = self.kernel_transforms[current_kernel_index](kernel_center)
            # the kernel has now advanced through the available sizes by one
            current_kernel = next_kernel
            current_kernel_index += 1

        return current_kernel

    def get_kernel(self):
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
            if all(self.out_channel_filter):
                # if out_channels are unfiltered, the output bias does not need filtering.
                return new_kernel, self.bias
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
        except ValueError:
            logging.warn(
                f"requested elastic dilation size {new_dilation_size} is not an available dilation size. Defaulting to full size ({self.max_dilation_size})"
            )

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
            logging.warn(
                f"selected kernel index {target_dilation_index} is out of range: 0 .. {len(self.dilation_sizes)}. Setting to last index."
            )
            target_dilation_index = len(self.dilation_sizes) - 1
        self.set_dilation_size(self.dilation_sizes[target_dilation_index])

    def get_available_dilation_steps(self):
        return len(self.dilation_sizes)

    def get_dilation_size(self):
        return self.dilation_sizes[self.target_dilation_index]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass
