import copy
from typing import List
import torch.nn as nn
import torch.nn.functional as nnf
import logging
import torch
from ..utilities import conv1d_get_padding, sub_filter_start_end


class ElasticKernelConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups : int = 1,
        bias: bool = False,
    ):
        # sort available kernel sizes from largest to smallest (descending order)
        kernel_sizes.sort(reverse=True)
        self.kernel_sizes: List[int] = kernel_sizes
        # after sorting kernel sizes, the maximum and minimum size available are the first and last element
        self.max_kernel_size: int = kernel_sizes[0]
        self.min_kernel_size: int = kernel_sizes[-1]
        # initially, the target size is the full kernel
        self.target_kernel_index: int = 0
        self.out_channels: int = out_channels
        # print(self.out_channels)
        super().__init__(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=self.max_kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # the list of kernel transforms will have one element less than the list of kernel sizes.
        # between every two sequential kernel sizes, there will be a kernel transform
        # the subsequent kernel is determined by applying the same-size center of the previous kernel to the transform
        self.kernel_transforms = nn.ModuleList([])
        for i in range(len(kernel_sizes) - 1):
            # the target size of the kernel transform is the next kernel size in the sequence
            new_kernel_size = kernel_sizes[i+1]
            # kernel transform is kept minimal by being shared between channels. It is simply a linear transformation from the center of the previous kernel to the new kernel
            # directly applying the kernel to the transform is possible: nn.Linear accepts multi-dimensional input in a way where the last input dim is transformed from in_channels to out_channels for the last output dim
            new_transform_module = nn.Linear(new_kernel_size, new_kernel_size, bias=False)
            # initialise the transform as the identity matrix to start training from the center of the larger kernel
            new_transform_module.weight.data.copy_(torch.eye(new_kernel_size))
            # transform weights are initially frozen
            new_transform_module.weight.requires_grad = False
            self.kernel_transforms.append(new_transform_module)
        self.set_kernel_size(self.max_kernel_size)

    def set_kernel_size(self, new_kernel_size):
        # previous_kernel_size = self.kernel_sizes[self.target_kernel_index]
        if new_kernel_size < self.min_kernel_size or new_kernel_size > self.max_kernel_size:
            logging.warn(f"requested elastic kernel size ({new_kernel_size}) outside of min/max range: ({self.max_kernel_size}, {self.min_kernel_size}). clamping.")
            if new_kernel_size < self.min_kernel_size:
                new_kernel_size = self.min_kernel_size
            else:
                new_kernel_size = self.max_kernel_size

        self.target_kernel_index = 0
        try:
            index = self.kernel_sizes.index(new_kernel_size)
            self.target_kernel_index = index
        except ValueError:
            logging.warn(f"requested elastic kernel size {new_kernel_size} is not an available kernel size. Defaulting to full size ({self.max_kernel_size})")

        # if the largest kernel is selected, train the actual kernel weights. For elastic sub-kernels, only the 'final' transform in the chain should be trained.
        if self.target_kernel_index == 0:
            self.weight.requires_grad = True
            for i in range(len(self.kernel_transforms)):
                # if the full kernel is selected for training, do not train the transforms
                self.kernel_transforms[i].weight.requires_grad = False
        else:
            self.weight.requires_grad = False
            for i in range(len(self.kernel_transforms)):
                # only the kernel transformation transforming to the current target index should be trained
                # the n-th transformation transforms the n-th kernel to the (n+1)-th kernel
                self.kernel_transforms[i].weight.requires_grad = i == (self.target_kernel_index - 1)
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
            logging.debug(f"unable to step down kernel size, no available index after current: {self.target_kernel_index} with size: {self.kernel_sizes[self.target_kernel_index]}")
            return False

    # lock/unlock training of the kernels
    # should only need to set requires_grad for the currently active kernel, the weights of other kernel sizes should be frozen
    def kernel_requires_grad(self, state: bool):
        if self.target_kernel_index == 0:
            self.weight.requires_grad = state
        else:
            # the (n-1)-th transform produces the weights of the n-th kernel from the (n-1)-th kernel.
            self.kernel_transforms[self.target_kernel_index-1].weight.requires_grad = state

    # TODO: while kernel transform size (and therefore the required computation) is almost negligible, the transformed second-to-last kernel could be cached
    # only the very last kernel transform should change unless the target kernel size changes.
    def get_kernel(self, in_channel=None):
        current_kernel_index = 0
        current_kernel = self.weight.data
        # for later: reduce channel count to first n channels
        if in_channel is not None:
            out_channel = in_channel
            current_kernel = current_kernel[:out_channel, :in_channel, :]
        # step through kernels until the target index is reached.
        while current_kernel_index < self.target_kernel_index:
            if current_kernel_index >= len(self.kernel_sizes):
                logging.warn(f"kernel size index {current_kernel_index} is out of range. Elastic kernel acquisition stopping at last available kernel")
                break
            # find start, end pos of the kernel center for the given next kernel size
            start, end = sub_filter_start_end(self.kernel_sizes[current_kernel_index], self.kernel_sizes[current_kernel_index+1])
            # extract the kernel center of the correct size
            kernel_center = current_kernel[:, :, start:end]
            # apply the kernel transformation to the next kernel. the n-th transformation is applied to the n-th kernel, yielding the (n+1)-th kernel
            next_kernel = self.kernel_transforms[current_kernel_index](kernel_center)
            # the kernel has now advanced through the available sizes by one
            current_kernel = next_kernel
            current_kernel_index += 1

        return current_kernel

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # return self.get_basic_conv1d().forward(input)  # for validaing assembled module
        # get the kernel for the current index
        kernel = self.get_kernel()
        # get padding for the size of the kernel
        padding = conv1d_get_padding(self.kernel_sizes[self.target_kernel_index])
        return nnf.conv1d(input, kernel, self.bias, self.stride, padding, self.dilation)

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_conv1d(self) -> nn.Conv1d:
        kernel = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        padding = conv1d_get_padding(kernel_size)
        new_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            bias=False
        )
        new_conv.weight.data = kernel
        new_conv.bias = self.bias
        # copy over elastic filter annotations if they are present
        elastic_width_filter_input = getattr(self, 'elastic_width_filter_input', None)
        elastic_width_filter_output = getattr(self, 'elastic_width_filter_output', None)
        if elastic_width_filter_input is not None:
            setattr(new_conv, 'elastic_width_filter_input', elastic_width_filter_input)
        if elastic_width_filter_output is not None:
            setattr(new_conv, 'elastic_width_filter_output', elastic_width_filter_output)
        # print("\nassembled a basic conv from elastic kernel!")
        return new_conv

    # return a safe copy of a conv1d equivalent to this module in the current state
    def assemble_basic_conv1d(self) -> nn.Conv1d:
        return copy.deepcopy(self.get_basic_conv1d())
