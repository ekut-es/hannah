import copy
from typing import List
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
import logging
import torch
# from ..utils import ConfigType, SerializableModule
from ..factory import qat as qat


# Conv1d with automatic padding for the set kernel size
def conv1d_auto_padding(conv1d: nn.Conv1d):
    conv1d.padding = conv1d_get_padding(conv1d.kernel_size[0])
    return conv1d


def conv1d_get_padding(kernel_size):
    padding = kernel_size // 2
    return padding


class ElasticKernelConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias: bool = False,
    ):
        # sort available kernel sizes from largest to smallest (descending order)
        kernel_sizes.sort(reverse=True)
        self.kernel_sizes = kernel_sizes
        # after sorting kernel sizes, the maximum and minimum size available are the first and last element
        self.max_kernel_size = kernel_sizes[0]
        self.min_kernel_size = kernel_sizes[-1]
        # initially, the target size is the full kernel
        self.target_kernel_index = 0
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
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
        # TODO: with dynamic channels this would use input.size(1) and pass it to get_kernel for the channel count
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
        # print("\nassembled a basic conv from elastic kernel!")
        return new_conv

    # return a safe copy of a conv1d equivalent to this module in the current state
    def assemble_basic_conv1d(self) -> nn.Conv1d:
        return copy.deepcopy(self.get_basic_conv1d())


# from ofa/utils/common_tools
def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


# base construct of a residual block
class ResBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, act_after_res=False, norm_after_res=False, norm_order=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.do_act = act_after_res
        self.do_norm = norm_after_res
        self.norm_order = norm_order
        # if the input channel count does not match the output channel count, apply skip to residual values
        self.apply_skip = self.in_channels != self.out_channels
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)
        # placeholders:
        self.blocks = nn.Identity()
        self.skip = nn.Identity()

    def forward(self, x):
        residual = x
        if self.apply_skip:
            residual = self.skip(residual)
        x = self.blocks(x)
        x += residual
        # do activation and norm after applying residual (if enabled)
        if self.do_norm and self.norm_order.norm_before_act:
            x = self.norm(x)
        if self.do_act:
            x = self.act(x)
        if self.do_norm and self.norm_order.norm_after_act:
            x = self.norm(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def get_nested_modules(self):
        return nn.ModuleList([
            self.blocks,
            self.skip,
            self.norm,
            self.act
        ])


# residual block with a 1d skip connection
class ResBlock1d(ResBlockBase):
    def __init__(self, in_channels, out_channels, minor_blocks, act_after_res=False, norm_after_res=False, stride=1, norm_order=None):
        super().__init__(in_channels=in_channels, out_channels=out_channels, act_after_res=act_after_res, norm_after_res=norm_after_res, norm_order=norm_order)
        # set the minor block sequence if specified in construction
        # if minor_blocks is not None:
        self.blocks = minor_blocks
        # if applying skip to the residual values is required, create skip as a minimal conv1d
        # stride is also applied to the skip layer (if specified, default is 1)
        self.skip = nn.Sequential(
            nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(self.out_channels)
        ) if self.apply_skip else None


def create(
    name: str,
    labels: int,
    input_shape,
    conv=[],
    min_depth: int = 1,
    norm_order=None,
    steps_without_sampling=1,
    steps_per_kernel_step=100
):
    # if no orders for the norm operator are specified, fall back to default
    if not (hasattr(norm_order, "norm_before_act") and hasattr(norm_order, "norm_after_act")):
        logging.info("order of norm before/after activation is not set!")
        norm_order = {"norm_before_act": True, "norm_after_act": False}

    flatten_n = input_shape[0]
    in_channels = input_shape[1]
    pool_n = input_shape[2]
    # the final output channel count is given by the last minor block of the last major block
    final_out_channels = conv[-1].blocks[-1].out_channels
    conv_layers = nn.ModuleList([])
    next_in_channels = in_channels

    for block_config in conv:
        if block_config.target == "forward":
            major_block = create_forward_block(blocks=block_config.blocks, in_channels=next_in_channels, stride=block_config.stride, norm_order=norm_order)
        elif block_config.target == "residual1d":
            major_block = create_residual_block_1d(blocks=block_config.blocks, in_channels=next_in_channels, stride=block_config.stride, norm_order=norm_order)
        else:
            raise Exception(f"Undefined target selected for major block: {block_config.target}")
        # output channel count of the last minor block will be the input channel count of the next major block
        next_in_channels = block_config.blocks[-1].out_channels
        conv_layers.append(major_block)

    # get the max depth from the count of major blocks
    model = WIPModel(
        conv_layers=conv_layers,
        max_depth=len(conv_layers),
        labels=labels,
        pool_kernel=pool_n,
        flatten_dims=flatten_n,
        out_channels=final_out_channels,
        min_depth=min_depth,
        block_config=conv,
        steps_without_sampling=steps_without_sampling,
        steps_per_kernel_step=steps_per_kernel_step
    )

    return model


# build a sequence from a list of minor block configurations
def create_minor_block_sequence(blocks, in_channels, stride=1, norm_order=None):
    next_in_channels = in_channels
    minor_block_sequence = nn.ModuleList([])
    is_first_minor_block = True
    for block_config in blocks:
        # set stride on the first minor block in the sequence
        if is_first_minor_block:
            next_stride = stride
            is_first_minor_block = False
        else:
            next_stride = 1
        minor_block, next_in_channels = create_minor_block(block_config=block_config, in_channels=next_in_channels, stride=next_stride, norm_order=norm_order)
        minor_block_sequence.append(minor_block)

    return module_list_to_module(minor_block_sequence)


# build a single minor block from its config. return the number of output channels with the block
def create_minor_block(block_config, in_channels: int, stride : int = 1, norm_order=None):
    new_block = None
    # the output channel count is usually stored in block_config.out_channels
    # use it as the default value if available, otherwise it must be set by the specific code handling the target type
    new_block_out_channels = getattr(block_config, "out_channels", 1)

    if block_config.target == "conv1d":
        out_channels = block_config.out_channels
        # create a conv minor block from the config, autoset padding
        minor_block_internal_sequence = nn.ModuleList([])
        new_minor_block = conv1d_auto_padding(nn.Conv1d(
                kernel_size=block_config.kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride
        ))
        minor_block_internal_sequence.append(new_minor_block)

        # add norm/act if requested
        norm_act_sequence = create_norm_act_sequence(block_config.norm, block_config.act, out_channels, norm_order)
        if norm_act_sequence is not None:
            minor_block_internal_sequence.append(norm_act_sequence)

        new_block = module_list_to_module(minor_block_internal_sequence)
        # the input channel count of the next minor block is the output channel count of the previous block
        new_block_out_channels = out_channels
    elif block_config.target == "elastic_conv1d":
        out_channels = block_config.out_channels
        kernel_sizes = block_config.kernel_sizes
        # create a minor block, potentially with activation and norm
        minor_block_internal_sequence = nn.ModuleList([])
        new_minor_block = ElasticKernelConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride
        )
        minor_block_internal_sequence.append(new_minor_block)

        # add norm/act if requested
        norm_act_sequence = create_norm_act_sequence(block_config.norm, block_config.act, out_channels, norm_order)
        if norm_act_sequence is not None:
            minor_block_internal_sequence.append(norm_act_sequence)

        new_block = module_list_to_module(minor_block_internal_sequence)
        # the input channel count of the next minor block is the output channel count of the previous block
        new_block_out_channels = out_channels
    # if an unknown target is selected for a minor block, throw an exception.
    else:
        raise Exception(f"Undefined target selected in minor block sequence: {block_config.target}")

    # return the new block and its output channel count
    return new_block, new_block_out_channels


# create a module representing a sequence of norm and act
def create_norm_act_sequence(norm: bool, act: bool, channels: int, norm_order=None):
    # batch norm will be added before and/or after activation depending on the configuration
    # fallback default is one norm before act, if no order is specified.

    # if no norm or activation is requested, simply return None
    # going through the steps below and returning an empty module list would also be fine
    if not norm and not act:
        return None

    if norm_order is None:
        norm_before_act = True
        norm_after_act = False
    else:
        norm_before_act = norm_order.norm_before_act
        norm_after_act = norm_order.norm_after_act

    norm_act_sequence = nn.ModuleList([])

    if norm and norm_before_act:
        norm_act_sequence.append(nn.BatchNorm1d(channels))
    if act:
        # add relu activation if act is set
        norm_act_sequence.append(nn.ReLU())
    if norm and norm_after_act:
        norm_act_sequence.append(nn.BatchNorm1d(channels))

    return module_list_to_module(norm_act_sequence)


# build a basic forward major block
def create_forward_block(blocks, in_channels, stride=1, norm_order=None):
    return create_minor_block_sequence(blocks, in_channels, stride=stride, norm_order=norm_order)


# build a residual major block
def create_residual_block_1d(blocks, in_channels, stride=1, norm_order=None):
    minor_blocks = create_minor_block_sequence(blocks, in_channels, stride=stride, norm_order=norm_order)
    # the output channel count of the residual major block is the output channel count of the last minor block
    out_channels = blocks[-1].out_channels
    residual_block = ResBlock1d(in_channels=in_channels, out_channels=out_channels, minor_blocks=minor_blocks, stride=stride, norm_order=norm_order)
    return residual_block


class WIPModel(nn.Module):
    def __init__(
        self, conv_layers: nn.ModuleList([]),
        max_depth: int,
        labels: int,
        pool_kernel: int,
        flatten_dims: int,
        out_channels: int,
        min_depth: int = 1,
        block_config=[],
        steps_without_sampling: int = 1,
        steps_per_kernel_step: int = 100
    ):
        super().__init__()
        self.conv_layers = conv_layers
        self.max_depth = max_depth
        self.active_depth = self.max_depth
        self.labels = labels
        self.pool_kernel = pool_kernel
        self.flatten_dims = flatten_dims
        self.out_channels = out_channels
        self.block_config = block_config
        self.min_depth = min_depth
        self.steps_without_sampling = steps_without_sampling
        self.steps_per_kernel_step = steps_per_kernel_step
        self.current_step = 0
        self.current_kernel_step = 0
        self.last_input = None
        # self.pool = nn.AvgPool1d(pool_kernel)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(flatten_dims)
        # one linear exit layer for each possible depth level
        self.linears = nn.ModuleList([])
        # for every possible depth level (from min_depth to including max_depth)
        for i in range(self.min_depth, self.max_depth+1):
            self.active_depth = i
            self.update_output_channel_count()
            # create the linear output layer for this depth
            self.linears.append(nn.Linear(self.out_channels, self.labels))
        # should now be redundant, as the loop will exit with the active depth being max_depth
        self.active_depth = self.max_depth

    def forward(self, x):
        self.last_input = x
        self.current_step = self.current_step + 1
        for layer in self.conv_layers[:self.active_depth]:
            x = layer(x)

        result = x
        # print(np.shape(result))
        result = self.pool(result)
        result = self.flatten(result)
        result = self.get_output_linear_layer(self.active_depth)(result)
        # print(np.shape(result))

        return result

    # return an extracted module sequence for a given depth
    def extract_elastic_depth_sequence(self, target_depth, quantized=False, clone_mode=False):
        if target_depth < self.min_depth or target_depth > self.max_depth:
            raise Exception(f"attempted to extract submodel for depth {target_depth} where min: {self.min_depth} and max: {self.max_depth}")
        extracted_module_list = nn.ModuleList([])

        if clone_mode:
            for layer in self.conv_layers[:target_depth]:
                extracted_module_list.append(layer)
        else:
            rebuild_output = rebuild_extracted_blocks(self.conv_layers[:target_depth], quantized=quantized)
            extracted_module_list.append(module_list_to_module(rebuild_output))
            # for item in rebuild_output:
            #     extracted_module_list.append(item)

        extracted_module_list.append(self.pool)
        extracted_module_list.append(self.flatten)
        extracted_module_list.append(self.get_output_linear_layer(target_depth))
        extracted_module_list = flatten_module_list(extracted_module_list)
        return copy.deepcopy(module_list_to_module(extracted_module_list))

    def get_elastic_depth_output(self, target_depth=None, quantized=False):
        if target_depth is None:
            target_depth = self.max_depth
        if self.last_input is None:
            return None
        submodel = self.extract_elastic_depth_sequence(target_depth, quantized=quantized)
        # print(submodel)
        # print(type(submodel))
        # print(self.last_input)
        # print(np.shape(self.last_input))
        output = submodel(self.last_input)
        # print(type(output))
        # print(output)
        return output

    # sample the active subnet, select a random depth between the configured min and the max depth (available major block depth)
    def sample_active_subnet(self):
        # only sample the subnet after the set amount of steps have passed
        if not self.should_subsample():
            return
        # active depth is picked from min depth to including max depth
        self.active_depth = np.random.randint(self.min_depth, self.max_depth+1)
        self.update_output_channel_count()
        # print("Picked active depth: ", self.active_depth)
        # also sample a random kernel, in place until progressive shrinking is done
        max_kernel_steps = self.compute_max_kernel_steps()
        # pick a random kernel step, from 0 to including the max step count
        random_kernel_step = np.random.randint(0, max_kernel_steps+1)
        self.go_to_kernel_step(random_kernel_step)

    def should_subsample(self, verify_step=False):
        # for testing, until there is a proper scheduler in place:
        self.check_kernel_stepping()
        # Shortcut for testing: set to True to also verify loss of equivalent extracted model
        if verify_step:
            return False
        return self.current_step > self.steps_without_sampling

    # placeholder until there is a proper scheduler in place
    def check_kernel_stepping(self):
        # if enough steps have passed, step down kernels by one size
        if self.current_step > self.steps_per_kernel_step*(self.current_kernel_step+1):
            # print("stepping kernels!")
            self.current_kernel_step += 1
            if not self.step_down_all_kernels():
                self.current_kernel_step -= 1
                setattr(self, 'max_kernel_steps', self.current_kernel_step)

        return self.current_step > self.steps_per_kernel_step

    # temporary implementation to speed up random sampling by only searching for max kernel steps once
    # get max amount of kernel steps
    def compute_max_kernel_steps(self):
        max_kernel_steps = getattr(self, 'max_kernel_steps', None)
        if max_kernel_steps is not None:
            # if the attribute was already set, simply return.
            return max_kernel_steps
        else:
            previous_kernel_step = self.current_kernel_step
            self.reset_all_kernel_sizes()
            # count the amount of steps until step_down_all_kernels returns false (step-down was no longer possible)
            steps_passed = 0
            while self.step_down_all_kernels():
                steps_passed += 1
            # store the value, it only needs to be computed once.
            setattr(self, 'max_kernel_steps', steps_passed)
            # return to whichever the current step was before this function was called
            self.go_to_kernel_step(previous_kernel_step)
            # return the computed value
            return steps_passed

    # reset elastic values to their default (max) values
    def reset_active_elastic_values(self):
        self.active_depth = self.max_depth
        # self.reset_all_kernel_sizes()

    # resume: return to the elastic values from before a reset
    def resume_active_elastic_values(self):
        self.resume_kernel_sizes_from_step()

    # set the output channel count value based on the current active depth
    def update_output_channel_count(self):
        # the new out channel count is given by the last minor block of the last active major block
        self.out_channels = self.block_config[:self.active_depth][-1].blocks[-1].out_channels

    # return the linear layer which processes the output for the current elastic depth
    def get_output_linear_layer(self, target_depth):
        return self.linears[target_depth-self.min_depth]

    # step all elastic kernels within the model down by one, if possible
    def step_down_all_kernels(self):
        return call_function_from_deep_nested(input=self.conv_layers, function="step_down_kernel_size", type_selection=ElasticKernelConv1d)
        # return call_function_from_deep_nested(input=self.conv_layers, function="step_down_kernel_size")

    # reset all kernel sizes to their max value
    def reset_all_kernel_sizes(self):
        return call_function_from_deep_nested(input=self.conv_layers, function="reset_kernel_size", type_selection=ElasticKernelConv1d)
        # return call_function_from_deep_nested(input=self.conv_layers, function="reset_kernel_size")

    # go to a specific kernel step
    def go_to_kernel_step(self, step: int):
        self.current_kernel_step = step
        self.resume_kernel_sizes_from_step()

    # go back to the kernel sizes specified by the current step
    # call after reset_all_kernel_sizes to resume
    def resume_kernel_sizes_from_step(self):
        # save the current step, resetting may also reset the value for some future implementations
        step = self.current_kernel_step
        # reset kernel sizes to start from a known point
        self.reset_all_kernel_sizes()
        self.current_kernel_step = step
        for i in range(self.current_kernel_step):
            # perform one step down call for each current kernel step
            if not self.step_down_all_kernels():
                # if this iteration of stepping down kernel size returned false, there were no kernels to step down. Further iterations are not necessary
                break


# recurse through any iterable (sub)structures. Attempt to call the specified function from any discovered objects if it is available.
# return true if any of the calls returned true
# for modules: both ModuleList and Sequential are iterable, so this should be able to descend into any module substructures
def call_function_from_deep_nested(input, function, type_selection : type = None):
    if input is None:
        return False
    # print(".")
    call_return_value = False
    # if a type is specified, only check matching objects
    if type_selection is None or isinstance(input, type_selection):
        # print(type(input))
        maybe_function = getattr(input, function, None)
        if callable(maybe_function):
            call_return_value = maybe_function()
            # print("deep call!")

    # if the input is iterable, recursively check any nested objects
    if hasattr(input, '__iter__'):
        for item in input:
            new_return_value = call_function_from_deep_nested(item, function, type_selection)
            call_return_value = call_return_value or new_return_value

    # if the object has a function to return nested modules, also check them.
    if callable(getattr(input, "get_nested_modules", None)):
        nested_modules = getattr(input, "get_nested_modules", None)()
        new_return_value = call_function_from_deep_nested(nested_modules, function, type_selection)
        call_return_value = call_return_value or new_return_value

    return call_return_value


def rebuild_extracted_blocks(blocks, quantized=False):
    out_modules = nn.ModuleList([])
    module_set = DefaultModuleSet1d()
    # print(f"\nRebuilding : {type(blocks)} {len(blocks)}")
    if quantized:
        module_set = QuantizedModuleSet1d()

    if blocks is None:
        raise ValueError("input blocks are None value")

    # if the input is not iterable, encase it in a moduleList
    elif not hasattr(blocks, '__iter__'):
        if not isinstance(blocks, nn.Module):
            raise TypeError("Input blocks are neither iterable nor Module")
        blocks = nn.ModuleList([blocks])

    if isinstance(blocks, nn.Sequential) or isinstance(blocks, nn.ModuleList):
        modules = nn.ModuleList([])
        for item in blocks:
            modules.append(item)

        modules = flatten_module_list(modules)

        input_modules_flat_length = len(modules)

        i = 0
        while i in range(len(modules)):
            module = modules[i]
            # print(type(module))
            reassembled_module = None

            # if the module is an elastic kernel convolution, it is replaced by an equivalent basic conv1d for its current state
            if isinstance(module, ElasticKernelConv1d):
                # print(type(module))
                replacement_module = module.assemble_basic_conv1d()
                module = replacement_module
                # print(type(module))

            if isinstance(module, nn.Conv1d):
                if i+1 in range(len(modules)) and isinstance(modules[i+1], nn.BatchNorm1d):
                    if i+2 in range(len(modules)) and isinstance(modules[i+2], nn.ReLU):
                        # if both norm and relu follow in sequence, combine all three and skip the next two items (which are the norm, act)
                        reassembled_module = module_set.reassemble(module=module, norm=True, act=True, norm_module=modules[i+1])
                        i += 2
                    else:
                        # if only norm follows in sequence, combine both and skip the next item (which is the norm)
                        reassembled_module = module_set.reassemble(module=module, norm=True, act=False, norm_module=modules[i+1])
                        i += 1
                elif i+1 in range(len(modules)) and isinstance(modules[i+1], nn.ReLU):
                    # if an act with no previous norm follows, combine both and skip the next item (which is the act)
                    reassembled_module = module_set.reassemble(module=module, norm=False, act=True)
                    i += 1
                else:
                    # if there is no norm or act after the conv, reassemble a standalone conv
                    reassembled_module = module_set.reassemble(module=module, norm=False, act=False)
            elif isinstance(module, nn.BatchNorm1d):
                if module_set.norm1d is not None:
                    # pass the channel count on to the new norm type
                    reassembled_module = module_set.norm1d(module.num_features)
                    reassembled_module.weight = module.weight
                else:
                    logging.error("Skipping stand-alone norm in reassembly: not available in the selected module set")
            elif isinstance(module, nn.ReLU):
                if module_set.act is not None:
                    reassembled_module = module_set.act()
                else:
                    logging.error("Skipping stand-alone activation in reassembly: not available in the selected module set")
            elif isinstance(module, ResBlockBase):
                # reassemble both the subblocks and the skip layer separately, then put them into a new ResBlock
                reassembled_subblocks = module_list_to_module(rebuild_extracted_blocks(module.blocks, quantized=quantized))
                reassembled_skip = module_list_to_module(rebuild_extracted_blocks(module.skip, quantized=quantized))
                reassembled_module = ResBlockBase(module.in_channels, module.out_channels)
                reassembled_module.blocks = reassembled_subblocks
                reassembled_module.skip = reassembled_skip

            # print(reassembled_module)
            if reassembled_module is not None:
                out_modules.append(reassembled_module)
            i += 1

    out_modules = flatten_module_list(out_modules)
    output_modules_flat_length = len(out_modules)
    if input_modules_flat_length != output_modules_flat_length and not quantized:
        logging.info("Reassembly changed length of module list")
    return out_modules


def flatten_module_list(modules):
    if not hasattr(modules, '__iter__'):
        if isinstance(modules, nn.Module):
            # if the input is non-iterable and is already a module, it can be returned as a list of one element
            return nn.ModuleList([modules])

    else:
        # flatten any nested Sequential or ModuleList
        contains_nested = (isinstance(x, nn.Sequential) for x in modules) or (isinstance(x, nn.ModuleList) for x in modules)
        # repeat until the cycle no longer finds nested modules
        while contains_nested:
            # print(f"Nested? {type(modules)} {len(modules)}")
            contains_nested = False
            new_module_list = nn.ModuleList([])
            for old_item in modules:
                if hasattr(old_item, '__iter__'):
                    contains_nested = True
                    for old_subitem in old_item:
                        new_module_list.append(old_subitem)
                else:
                    new_module_list.append(old_item)
            modules = new_module_list

        return modules


# return a single module from an input moduleList
def module_list_to_module(module_list):
    # if the input is a Sequential module it will be iterable, but can be returned as is.
    if isinstance(module_list, nn.Sequential):
        return module_list
    # if the input is not already a module, it must be iterable
    if not hasattr(module_list, '__iter__'):
        if isinstance(module_list, nn.Module):
            return module_list
        raise TypeError("input is neither iterable nor module")
    if len(module_list) == 1:
        module = module_list[0]
        assert isinstance(module, nn.Module), "Iterable single-length input does not contain module"
        return module
    else:
        return nn.Sequential(*module_list)


class ModuleSet():
    conv1d = None
    norm1d = None
    act = None
    conv1d_norm_act = None
    conv1d_norm = None
    conv1d_act = None

    def reassemble(self, weights, norm, act, norm_module):
        raise Exception("reassemble function of module set is undefined")


class DefaultModuleSet1d(ModuleSet):
    conv1d = nn.Conv1d
    norm1d = nn.BatchNorm1d
    act = nn.ReLU

    def reassemble(self, module: nn.Conv1d, norm=False, act=False, norm_module: nn.BatchNorm1d = None, clone_conv=False, clone_norm=False):
        modules = nn.ModuleList([])
        # create a new conv1d under the same parameters, with the same weights
        # this could technically re-use the input module directly, as nothing is changed
        module = copy.deepcopy(module)
        norm_module = copy.deepcopy(norm_module)
        new_conv = module
        if not clone_conv:
            new_conv = self.conv1d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding
                )
            new_conv.weight = module.weight
            new_conv.bias = module.bias
        modules.append(new_conv)
        if norm:
            if norm_module is None:
                raise ValueError("reassembly with norm requested, no source norm module provided")
            new_norm = norm_module
            if not clone_norm:
                new_norm = self.norm1d(norm_module.num_features)
                new_norm.weight = norm_module.weight
                new_norm.bias = norm_module.bias
                new_norm.running_mean = norm_module.running_mean
                new_norm.running_var = norm_module.running_var
                new_norm.num_batches_tracked = norm_module.num_batches_tracked
                new_norm.eps = norm_module.eps
                new_norm.momentum = norm_module.momentum
                new_norm.eval()
            modules.append(new_norm)
        if act:
            modules.append(self.act())
        return copy.deepcopy(nn.Sequential(*modules))


# TODO: verify functionality (weight copying from normal to quantized)
class QuantizedModuleSet1d(ModuleSet):
    conv1d = qat.Conv1d
    conv1d_norm_act = qat.ConvBnReLU1d
    conv1d_norm = qat.ConvBn1d
    conv1d_act = qat.ConvReLU1d

    # TODO: copy norm weights?
    def reassemble(self, module: nn.Conv1d, norm=False, act=False, norm_module: nn.BatchNorm1d = None):
        out_module = None
        if norm and norm_module is None:
            raise ValueError("module with norm requested, no source norm module provided")
        if norm and act:
            # out_module = self.conv1d_norm_act(in_channels=module.in_channels, out_channels=module.out_channels, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups, bias=module.bias, padding_mode=module.padding_mode)
            out_module = self.conv1d_norm_act(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding)
            out_module.bn
        elif norm:
            # out_module = self.conv1d_norm(in_channels=module.in_channels, out_channels=module.out_channels, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups, bias=module.bias, padding_mode=module.padding_mode)
            out_module = self.conv1d_norm(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding)
        elif act:
            # out_module = self.conv1d_act(in_channels=module.in_channels, out_channels=module.out_channels, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups, bias=module.bias, padding_mode=module.padding_mode)
            out_module = self.conv1d_act(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding)
        else:
            # out_module = self.conv1d(in_channels=module.in_channels, out_channels=module.out_channels, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups, bias=module.bias, padding_mode=module.padding_mode)
            out_module = self.conv1d(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding)
        out_module.weight = module.weight
        out_module.bias = module.bias
        return copy.deepcopy(out_module)
