import copy
from typing import List, Tuple
import torch.nn as nn

# import torch.nn.functional as nnf
import numpy as np
import logging
import torch

# from ..utils import ConfigType, SerializableModule
from ..factory import qat as qat
from .submodules.elasticchannelhelper import ElasticChannelHelper
from .submodules.elastickernelconv import ElasticKernelConv1d
from .submodules.resblock import ResBlock1d, ResBlockBase
from .utilities import (
    flatten_module_list,
    get_instances_from_deep_nested,
    # set_basic_weight_grad,
    module_list_to_module,
    conv1d_auto_padding,
    call_function_from_deep_nested,
    # set_weight_maybe_bias_grad,
)


def create(
    name: str, labels: int, input_shape, conv=[], min_depth: int = 1, norm_order=None
) -> nn.Module:
    # if no orders for the norm operator are specified, fall back to default
    if not (
        hasattr(norm_order, "norm_before_act") or hasattr(norm_order, "norm_after_act")
    ):
        logging.info("order of norm before/after activation is not set!")
        norm_order = {"norm_before_act": True, "norm_after_act": False}

    flatten_n = input_shape[0]
    in_channels = input_shape[1]
    pool_n = input_shape[2]
    # the final output channel count is given by the last minor block of the last major block
    final_out_channels = conv[-1].blocks[-1].out_channels
    if hasattr(final_out_channels, "__iter__"):
        # if the output channel count is a list, get the highest value
        final_out_channels = max(final_out_channels)
    conv_layers = nn.ModuleList([])
    next_in_channels = in_channels

    previous_sources = [nn.ModuleList([])]
    previous_elastic_channel_helper: ElasticChannelHelper = None
    for block_config in conv:
        if block_config.target == "forward":
            major_block = create_forward_block(
                blocks=block_config.blocks,
                in_channels=next_in_channels,
                stride=block_config.stride,
                norm_order=norm_order,
                sources=previous_sources,
            )
            # this major block is the source for the next block.
            # If it already ends in a channel helper and is passed to a second
            # channel helper, the issue will be handled there.
            previous_sources = [major_block]
            if previous_elastic_channel_helper is not None:
                # if an elastic channel helper directly precedes this block,
                # this block is it's primary target.
                previous_elastic_channel_helper.set_primary_target(major_block)
            if hasattr(major_block, "__iter__") and isinstance(
                major_block[-1], ElasticChannelHelper
            ):
                # if the block ends in an elastic channel helper, store it.
                # It's targets are specified by the block which follows
                previous_elastic_channel_helper = major_block[-1]
        elif block_config.target == "residual1d":
            major_block = create_residual_block_1d(
                blocks=block_config.blocks,
                in_channels=next_in_channels,
                stride=block_config.stride,
                norm_order=norm_order,
                sources=previous_sources,
            )
            # this major block is the source for the next block.
            # This involves both the main blocks and the skip connection
            # both sources MUST be processed in parallel, and not added to one ModuleList:
            # each source is only ascended until a primary target is found.
            # Modules before it must be unaffected.
            # duplicate elastic channel helpers will be found if the source is added to the second helper module.
            previous_sources = [major_block.blocks, major_block.skip]
            if previous_elastic_channel_helper is not None:
                # if an elastic channel helper directly precedes this block, this block is it's primary target.
                previous_elastic_channel_helper.set_primary_target(major_block.blocks)
                # the input channels of the skip connection must also be modified by the channel helper
                previous_elastic_channel_helper.add_secondary_targets(major_block.skip)
            if hasattr(major_block.blocks, "__iter__") and isinstance(
                major_block.blocks[-1], ElasticChannelHelper
            ):
                helper: ElasticChannelHelper = major_block.blocks[-1]
                # if the block ends in an elastic channel helper, store it.
                # It's targets are specified by the block which follows
                previous_elastic_channel_helper = helper
                # additionally, this channel helper must be able to adjust the
                # output channels of the skip connection, if the residual block adjusts it's output channels
                helper.add_sources(major_block.skip)
        else:
            raise Exception(
                f"Undefined target selected for major block: {block_config.target}"
            )
        # output channel count of the last minor block will be the input channel count of the next major block
        next_in_channels = block_config.blocks[-1].out_channels
        if hasattr(next_in_channels, "__iter__"):
            # if the channel count is a list, get the highest value
            next_in_channels = max(next_in_channels)
        conv_layers.append(major_block)

    # get the max depth from the count of major blocks
    model = OFAModel(
        conv_layers=conv_layers,
        max_depth=len(conv_layers),
        labels=labels,
        pool_kernel=pool_n,
        flatten_dims=flatten_n,
        out_channels=final_out_channels,
        min_depth=min_depth,
        block_config=conv,
    )

    # store the name onto the model
    setattr(model, "creation_name", name)

    # acquire step counts for OFA progressive shrinking
    ofa_steps_depth = len(model.linears)
    ofa_steps_kernel = 1
    ofa_steps_width = 1
    for major_block in conv:
        for block in major_block.blocks:
            if block.target == "elastic_conv1d":
                this_block_kernel_steps = len(block.kernel_sizes)
                this_block_width_steps = len(block.out_channels)
                ofa_steps_width = max(ofa_steps_width, this_block_width_steps)
                ofa_steps_kernel = max(ofa_steps_kernel, this_block_kernel_steps)
            elif block.target == "elastic_channel_helper":
                this_block_width_steps = len(block.out_channels)
                ofa_steps_width = max(ofa_steps_width, this_block_width_steps)
    logging.info(
        f"OFA steps are {ofa_steps_kernel} kernel sizes, {ofa_steps_depth} depths, {ofa_steps_width} widths."
    )
    model.ofa_steps_kernel = ofa_steps_kernel
    model.ofa_steps_depth = ofa_steps_depth
    model.ofa_steps_width = ofa_steps_width

    return model


# build a sequence from a list of minor block configurations
def create_minor_block_sequence(
    blocks,
    in_channels,
    stride=1,
    norm_order=None,
    sources: List[nn.Module] = [nn.ModuleList([])],
) -> nn.Module:
    next_in_channels = in_channels
    minor_block_sequence = nn.ModuleList([])
    is_first_minor_block = True
    elastic_helper = None
    for block_config in blocks:
        # set stride on the first minor block in the sequence
        if is_first_minor_block:
            next_stride = stride
            is_first_minor_block = False
        else:
            next_stride = 1
        minor_block, next_in_channels = create_minor_block(
            block_config=block_config,
            in_channels=next_in_channels,
            stride=next_stride,
            norm_order=norm_order,
            sources=sources,
        )
        if hasattr(minor_block, "__iter__") and isinstance(
            minor_block[-1], ElasticChannelHelper
        ):
            # if the minor block is iterable, check it's last element for an elastic width helper.
            # It will already know it's source modules from the block
            # store the elastic helper, it needs to know it's 'target' blocks, which follow it
            elastic_helper = minor_block[-1]
            # reset sources: this block is already attached to an elastic channel helper
            sources = [nn.ModuleList([])]
        elif isinstance(minor_block, ElasticChannelHelper):
            # if the module is a standalone elastic channel helper, it has
            # already received the sources, but must still be stored for passing it's targets
            elastic_helper = minor_block
            # reset sources: no additional elastic channel helper may be attached to this block
            sources = [nn.ModuleList([])]
        else:
            # if the module does not end in any elastic helper, pass it as the target to a previous helper module, if present.
            if elastic_helper is not None:
                elastic_helper.set_primary_target(minor_block)
            # reset any previous stored helper if this block does not contain one
            elastic_helper = None
            # the minor block will be a source for the next minor block
            sources = minor_block

        minor_block_sequence.append(minor_block)

    return module_list_to_module(minor_block_sequence)


# build a single minor block from its config. return the number of output channels with the block
def create_minor_block(
    block_config,
    in_channels: int,
    stride: int = 1,
    norm_order=None,
    sources: List[nn.ModuleList] = [nn.ModuleList([])],
) -> Tuple[nn.Module, int]:
    new_block = None
    # the output channel count is usually stored in block_config.out_channels
    # use it as the default value if available, otherwise it must be set by the specific code handling the target type
    new_block_out_channels = getattr(block_config, "out_channels", 1)

    if block_config.target == "conv1d":
        out_channels = block_config.out_channels
        # create a conv minor block from the config, autoset padding
        minor_block_internal_sequence = nn.ModuleList([])
        new_minor_block = conv1d_auto_padding(
            nn.Conv1d(
                kernel_size=block_config.kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
            )
        )
        minor_block_internal_sequence.append(new_minor_block)

        # add norm/act if requested
        add_norm = block_config.get("norm", False)
        add_act = block_config.get("act", False)
        norm_act_sequence = create_norm_act_sequence(
            add_norm, add_act, out_channels, norm_order
        )

        if norm_act_sequence is not None:
            minor_block_internal_sequence.append(norm_act_sequence)

        new_block = module_list_to_module(
            flatten_module_list(minor_block_internal_sequence)
        )
        # the input channel count of the next minor block is the output channel count of the previous block
        new_block_out_channels = out_channels
    elif block_config.target == "elastic_conv1d":
        out_channels_list = block_config.out_channels
        out_channels_list.sort(reverse=True)
        # the maximum available width is the initial output channel count
        out_channels_full = out_channels_list[0]
        kernel_sizes = block_config.kernel_sizes
        # create a minor block, potentially with activation and norm
        minor_block_internal_sequence = nn.ModuleList([])
        new_minor_block = ElasticKernelConv1d(
            in_channels=in_channels,
            out_channels=out_channels_full,
            kernel_sizes=kernel_sizes,
            stride=stride,
        )
        minor_block_internal_sequence.append(new_minor_block)

        # add norm/act if requested
        norm_act_sequence = create_norm_act_sequence(
            block_config.norm, block_config.act, out_channels_full, norm_order
        )
        if norm_act_sequence is not None:
            minor_block_internal_sequence.append(norm_act_sequence)

        new_block = module_list_to_module(
            flatten_module_list(minor_block_internal_sequence)
        )
        # if multiple output channel widths are specified (elastic width), add an elastic width helper module
        if len(out_channels_list) > 1:
            # the sources of the elastic channel helper module are the previous conv, and its potential norm/act
            helper_module = ElasticChannelHelper(out_channels_list, new_block, None)
            # append the helper module to the sequence
            new_sequence = nn.ModuleList([new_block, helper_module])
            new_block = module_list_to_module(new_sequence)
        # the input channel count of the next minor block is the output channel count of the previous block
        # output channel count is specified by the elastic conv
        new_block_out_channels = new_minor_block.out_channels
    elif block_config.target == "elastic_channel_helper":
        # if the module is a standalone elastic channel helper, pass the previous block as it's sources
        out_channels_list = block_config.out_channels
        out_channels_list.sort(reverse=True)
        out_channels_full = out_channels_list[0]
        new_block = ElasticChannelHelper(out_channels_list, None, None)
        for source in sources:
            # add every source item as a source to the new block.
            # this has to be done in parallel: each source is only ascended
            # until a primary target is found (modules before it must be unaffected.)
            new_block.add_sources(source)
        if out_channels_full != in_channels:
            logging.error(
                f"standalone ElasticChannelHelper input width {in_channels} does not match max output channel width {out_channels_full} in list {out_channels_list}"
            )
        new_block_out_channels = out_channels_full
    # if an unknown target is selected for a minor block, throw an exception.
    else:
        raise Exception(
            f"Undefined target selected in minor block sequence: {block_config.target}"
        )

    # return the new block and its output channel count
    return new_block, new_block_out_channels


# create a module representing a sequence of norm and act
def create_norm_act_sequence(
    norm: bool, act: bool, channels: int, norm_order=None
) -> nn.Module:
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
    # create the norm module only if required. its reference will be passed back.
    new_norm = None
    if norm:
        new_norm = nn.BatchNorm1d(channels)
    new_act = nn.ReLU()
    if norm and norm_before_act:
        norm_act_sequence.append(new_norm)
    if act:
        # add relu activation if act is set
        norm_act_sequence.append(new_act)
    if norm and norm_after_act:
        norm_act_sequence.append(norm)

    return module_list_to_module(norm_act_sequence)


# build a basic forward major block
def create_forward_block(
    blocks,
    in_channels,
    stride=1,
    norm_order=None,
    sources: List[nn.ModuleList] = [nn.ModuleList([])],
) -> nn.Module:
    return create_minor_block_sequence(
        blocks, in_channels, stride=stride, norm_order=norm_order, sources=sources
    )


# build a residual major block
def create_residual_block_1d(
    blocks,
    in_channels,
    stride=1,
    norm_order=None,
    sources: List[nn.ModuleList] = [nn.ModuleList([])],
) -> ResBlock1d:
    minor_blocks = create_minor_block_sequence(
        blocks, in_channels, stride=stride, norm_order=norm_order, sources=sources
    )
    # the output channel count of the residual major block is the output channel count of the last minor block
    out_channels = blocks[-1].out_channels
    if hasattr(out_channels, "__iter__"):
        # if the out_channels count is a list, get the highest value
        out_channels = max(out_channels)
    residual_block = ResBlock1d(
        in_channels=in_channels,
        out_channels=out_channels,
        minor_blocks=minor_blocks,
        stride=stride,
        norm_order=norm_order,
    )
    return residual_block


class OFAModel(nn.Module):
    def __init__(
        self,
        conv_layers: nn.ModuleList([]),
        max_depth: int,
        labels: int,
        pool_kernel: int,
        flatten_dims: int,
        out_channels: int,
        min_depth: int = 1,
        block_config=[],
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
        self.current_step = 0
        self.current_kernel_step = 0
        self.current_channel_step = 0
        self.current_width_step = 0
        self.sampling_max_kernel_step = 0
        self.sampling_max_depth_step = 0
        self.eval_mode = False
        self.last_input = None
        # will be updated with the output channel count
        self.active_elastic_output_helper: ElasticChannelHelper = None
        # self.pool = nn.AvgPool1d(pool_kernel)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(flatten_dims)
        # one linear exit layer for each possible depth level
        self.linears = nn.ModuleList([])
        # for every possible depth level (from min_depth to including max_depth)
        for i in range(self.min_depth, self.max_depth + 1):
            self.active_depth = i
            self.update_output_channel_count()
            # create the linear output layer for this depth
            new_output_linear = nn.Linear(self.out_channels, self.labels)
            if self.active_elastic_output_helper is not None:
                # add this output linear as a target to an elastic channel module
                # preceding it. The in-channels of this linear will also need to be modified.
                self.active_elastic_output_helper.add_secondary_targets(
                    new_output_linear
                )
            self.linears.append(new_output_linear)
        # should now be redundant, as the loop will exit with the active depth being max_depth
        self.active_depth = self.max_depth
        # ofa step counts will be set by the create function.
        self.ofa_steps_kernel = 1
        self.ofa_steps_depth = 1
        self.ofa_steps_width = 1
        # create a list of every elastic kernel conv, for sampling
        self.elastic_kernel_convs = get_instances_from_deep_nested(input=self.conv_layers, type_selection=ElasticKernelConv1d)
        logging.info(f"OFA model accumulated {len(self.elastic_kernel_convs)} elastic kernel convolutions for sampling.")

    def forward(self, x):
        self.last_input = x
        self.current_step = self.current_step + 1
        # if the network is currently being evaluated, don't sample a subnetwork!
        if (self.sampling_max_depth_step > 0 or self.sampling_max_kernel_step > 0) and not self.eval_mode:
            self.sample_subnetwork()
        for layer in self.conv_layers[: self.active_depth]:
            x = layer(x)

        result = x
        result = self.pool(result)
        result = self.flatten(result)
        result = self.get_output_linear_layer(self.active_depth)(result)

        return result

    # pick a random subnetwork, return the settings used
    def sample_subnetwork(self):
        state = {"depth_step": 0, "kernel_steps": []}
        new_depth_step = np.random.randint(self.sampling_max_depth_step+1)
        self.active_depth = self.max_depth - new_depth_step
        state["depth_step"] = new_depth_step
        # this would step every kernel the same amount
        # new_kernel_step = np.random.randint(self.sampling_max_kernel_step + 1)
        # self.go_to_kernel_step(new_kernel_step)
        for conv in self.elastic_kernel_convs:
            # pick an available kernel index for every elastic kernel conv, independently.
            max_available_sampling_step = min(self.sampling_max_kernel_step, conv.get_available_kernel_steps())
            new_kernel_step = np.random.randint(max_available_sampling_step+1)
            conv.pick_kernel_index(new_kernel_step)
            state["kernel_steps"].append(new_kernel_step)
        print(state)
        return state

    # return an extracted module sequence for a given depth
    def extract_elastic_depth_sequence(
        self, target_depth, quantized=False, clone_mode=False
    ):
        if target_depth < self.min_depth or target_depth > self.max_depth:
            raise Exception(
                f"attempted to extract submodel for depth {target_depth} where min: {self.min_depth} and max: {self.max_depth}"
            )
        extracted_module_list = nn.ModuleList([])

        if clone_mode:
            for layer in self.conv_layers[:target_depth]:
                extracted_module_list.append(layer)
        else:
            rebuild_output = rebuild_extracted_blocks(
                self.conv_layers[:target_depth], quantized=quantized
            )
            extracted_module_list.append(module_list_to_module(rebuild_output))

        extracted_module_list.append(self.pool)
        extracted_module_list.append(self.flatten)
        output_linear = self.get_output_linear_layer(target_depth)
        # apply potential channel filters of the output linear
        output_linear = apply_channel_filters(output_linear)
        extracted_module_list.append(output_linear)
        extracted_module_list = flatten_module_list(extracted_module_list)
        return copy.deepcopy(module_list_to_module(extracted_module_list))

    # return extracted module for a given progressive shrinking depth step
    def extract_module_from_depth_step(self, depth_step) -> nn.Module:
        torch_module = self.extract_elastic_depth_sequence(self.max_depth - depth_step)
        return torch_module

    def get_elastic_depth_output(self, target_depth=None, quantized=False):
        if target_depth is None:
            target_depth = self.max_depth
        if self.last_input is None:
            return None
        submodel = self.extract_elastic_depth_sequence(
            target_depth, quantized=quantized
        )
        output = submodel(self.last_input)
        return output

    # step all input widths within the model down by one, if possible
    def step_down_all_channels(self):
        # print("stepping down input widths by one!")
        self.current_width_step += 1
        if (self.current_width_step > self.ofa_steps_width):
            logging.warn(
                f"excessive OFA width stepping! Attempting to step down width when step limit {self.ofa_steps_width} already reached."
            )
        return call_function_from_deep_nested(
            input=self.conv_layers,
            function="step_down_input_width",
            type_selection=ElasticKernelConv1d,
        )

    def reset_active_depth(self):
        self.active_depth = self.max_depth

    # resume: return to the elastic values from before a reset
    def resume_active_elastic_values(self):
        self.resume_kernel_sizes_from_step()

    # set the output channel count value based on the current active depth
    def update_output_channel_count(self):
        # the new out channel count is given by the last minor block of the last active major block
        last_active_major_block = self.block_config[: self.active_depth][-1].blocks[-1]
        self.out_channels = last_active_major_block.out_channels
        # for error reporting below
        out_channels_maybe_list = self.out_channels
        if hasattr(self.out_channels, "__iter__"):
            # if the out_channels count is a list, get the highest value
            self.out_channels = max(self.out_channels)
            # get the very last module of the last active layer. It must be an elastic channel helper, as the channel count is a list.
            last_active_item = self.conv_layers[: self.active_depth][-1]
            if isinstance(last_active_item, ResBlock1d):
                # incase of a residual layer being at the end, the helper will be at the end of its blocks
                last_active_item = last_active_item.blocks
            if hasattr(last_active_item, "__iter__"):
                # flatten the iterable list of items to actually access the last module, and not some nested Sequential
                last_active_item = flatten_module_list(last_active_item)
                # a layer ususally contains multiple modules and is iterable.
                # Pick the last module within the layer.
                last_active_item = last_active_item[-1]
            # store the ElasticChannelHelper ending the active layer.
            # The correct output layer must be added as a secondary target.
            if isinstance(last_active_item, ElasticChannelHelper):
                self.active_elastic_output_helper = last_active_item
            else:
                logging.error(
                    f"model layer ends with multiple possible output channels {out_channels_maybe_list}, but last module '{last_active_item}' is not ElasticChannelHelper"
                )
        else:
            self.active_elastic_output_helper = None

    # return the linear layer which processes the output for the current elastic depth
    def get_output_linear_layer(self, target_depth):
        return self.linears[target_depth - self.min_depth]

    # step all elastic kernels within the model down by one, if possible
    def step_down_all_kernels(self):
        return call_function_from_deep_nested(
            input=self.conv_layers,
            function="step_down_kernel_size",
            type_selection=ElasticKernelConv1d,
        )
        # return call_function_from_deep_nested(input=self.conv_layers, function="step_down_kernel_size")

    # reset all kernel sizes to their max value
    def reset_all_kernel_sizes(self):
        return call_function_from_deep_nested(
            input=self.conv_layers,
            function="reset_kernel_size",
            type_selection=ElasticKernelConv1d,
        )
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
                # if this iteration of stepping down kernel size returned false,
                # there were no kernels to step down. Further iterations are not necessary
                break

    """
    # step active depth down by one. Freeze output weights of the previous depth step (now no longer in use)
    def step_active_depth(self):
        previous_output_linear = self.get_output_linear_layer(self.active_depth)
        set_basic_weight_grad(previous_output_linear, False)
        if self.active_depth > self.min_depth:
            self.active_depth -= 1
        else:
            logging.warn(
                f"Excess OFA depth stepping: step_active_depth called when min depth ({self.min_depth}) was already reached!"
            )

    # freeze 'normal' weights of modules. To be called after warm-up
    # this will freeze: conv weights (not elastic kernel transforms), batchnorm weights, linear weights
    def freeze_basic_module_weights(self):
        set_basic_weight_grad(self.conv_layers, False)

    # freeze all kernel weights of elastic kernel modules - both full kernels and kernel transforms.
    # to be called after the elastic kernel training step has completed.
    def freeze_elastic_kernels(self):
        call_function_from_deep_nested(
            input=self.conv_layers,
            function="freeze_kernel_weights",
            type_selection=ElasticKernelConv1d,
        )

    # freeze the weights of the full-depth output linear.
    # To be called after initial warm-up period (before elastic kernel training).
    def freeze_full_depth_linear(self):
        set_weight_maybe_bias_grad(self.linears[-1], False)

    # unfreeze elastic depth output layers (the full depth output layer weights remain frozen).
    def unfreeze_elastic_depths(self):
        for linear in self.linears[:-1]:
            set_weight_maybe_bias_grad(linear, True)

    # freeze weights of all output layers.
    # To be called after elastic depth training is completed.
    def freeze_all_depths(self):
        for linear in self.linears:
            set_weight_maybe_bias_grad(linear, False)

    def unfreeze_all_depths(self):
        for linear in self.linears:
            set_weight_maybe_bias_grad(linear, True)

    # called when warmup is completed and elastic kernel training should start.
    def progressive_shrinking_from_warmup_to_kernel(self):
        self.freeze_basic_module_weights()
        self.freeze_full_depth_linear()

    # called to perform one kernel step.
    def progressive_shrinking_kernel_step(self):
        self.step_down_all_kernels()

    # called when elastic kernel training is completed and elastic depth training should start.
    def progressive_shrinking_from_kernel_to_depth(self):
        self.reset_all_kernel_sizes()
        # setting kernel size to full can unfreeze the base conv module kernel weights.
        self.freeze_basic_module_weights()
        self.freeze_elastic_kernels()
        # (technically not required: elastic depth output linears should not be frozen at this point in time.)
        self.unfreeze_elastic_depths()

    # called to perform one depth step.
    def progressive_shrinking_perform_depth_step(self):
        self.step_active_depth()

    # called when elastic depth training is completed and elastic width training should start.
    def progressive_shrinking_from_depth_to_width(self):
        self.reset_active_depth()
        self.freeze_all_depths()

    # called to perform one width step.
    def progressive_shrinking_perform_width_step(self):
        self.step_down_all_channels()

    # restart all elastic values, except for the width
    def progressive_shrinking_restart_non_width(self):
        set_basic_weight_grad(self.conv_layers, True)
        self.reset_all_kernel_sizes()
        self.unfreeze_all_depths()
        self.reset_active_depth()
    """

    def progressive_shrinking_perform_width_step(self):
        self.step_down_all_channels()

    def progressive_shrinking_add_kernel(self):
        self.sampling_max_kernel_step += 1
        if self.sampling_max_kernel_step >= self.ofa_steps_kernel:
            self.sampling_max_kernel_step -= 1
            logging.warn(
                f"excessive OFA kernel stepping! Attempting to add a kernel step when max ({self.ofa_steps_kernel}) already reached"
            )

    def progressive_shrinking_add_depth(self):
        self.sampling_max_depth_step += 1
        if self.sampling_max_depth_step >= self.ofa_steps_depth:
            self.sampling_max_depth_step -= 1
            logging.warn(
                f"excessive OFA depth stepping! Attempting to add a depth step when max ({self.ofa_steps_kernel}) already reached"
            )

    def progressive_shrinking_disable_sampling(self):
        self.sampling_max_kernel_step = 0
        self.sampling_max_depth_step = 0


def rebuild_extracted_blocks(blocks, quantized=False):
    out_modules = nn.ModuleList([])
    module_set = DefaultModuleSet1d()
    if quantized:
        module_set = QuantizedModuleSet1d()

    if blocks is None:
        raise ValueError("input blocks are None value")

    # if the input is not iterable, encase it in a moduleList
    elif not hasattr(blocks, "__iter__"):
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
            reassembled_module = None

            # if the module is an elastic kernel convolution, it is replaced by an equivalent basic conv1d for its current state
            if isinstance(module, ElasticKernelConv1d):
                replacement_module = module.assemble_basic_conv1d()
                # assemble_basic_conv1d copies over the elastic filter information
                module = replacement_module

            if isinstance(module, nn.Conv1d):
                # apply channel filters to the conv, if present.
                module = apply_channel_filters(module)
                if i + 1 in range(len(modules)) and isinstance(
                    modules[i + 1], nn.BatchNorm1d
                ):
                    # apply channel filters to the norm, if present.
                    norm_module = apply_channel_filters(modules[i + 1])
                    if i + 2 in range(len(modules)) and isinstance(
                        modules[i + 2], nn.ReLU
                    ):
                        # if both norm and relu follow in sequence,
                        # combine all three and skip the next two items (which are the norm, act)
                        reassembled_module = module_set.reassemble(
                            module=module, norm=True, act=True, norm_module=norm_module
                        )
                        i += 2
                    else:
                        # if only norm follows in sequence,
                        # combine both and skip the next item (which is the norm)
                        reassembled_module = module_set.reassemble(
                            module=module, norm=True, act=False, norm_module=norm_module
                        )
                        i += 1
                elif i + 1 in range(len(modules)) and isinstance(
                    modules[i + 1], nn.ReLU
                ):
                    # if an act with no previous norm follows,
                    # combine both and skip the next item (which is the act)
                    reassembled_module = module_set.reassemble(
                        module=module, norm=False, act=True
                    )
                    i += 1
                else:
                    # if there is no norm or act after the conv,
                    # reassemble a standalone conv
                    reassembled_module = module_set.reassemble(
                        module=module, norm=False, act=False
                    )
            elif isinstance(module, nn.BatchNorm1d):
                # for standalone batchnorms, apply any channel filters, if present.
                module = apply_channel_filters(module)
                if module_set.norm1d is not None:
                    # pass the channel count on to the new norm type
                    reassembled_module = module_set.norm1d(module.num_features)
                    reassembled_module.weight = module.weight
                else:
                    logging.error(
                        "Skipping stand-alone norm in reassembly: not available in the selected module set"
                    )
            elif isinstance(module, nn.ReLU):
                if module_set.act is not None:
                    reassembled_module = module_set.act()
                else:
                    logging.error(
                        "Skipping stand-alone activation in reassembly: not available in the selected module set"
                    )
            elif isinstance(module, ResBlockBase):
                # reassemble both the subblocks and the skip layer separately, then put them into a new ResBlock
                reassembled_subblocks = module_list_to_module(
                    rebuild_extracted_blocks(module.blocks, quantized=quantized)
                )
                reassembled_skip = module_list_to_module(
                    rebuild_extracted_blocks(module.skip, quantized=quantized)
                )
                reassembled_module = ResBlockBase(
                    module.in_channels, module.out_channels
                )
                reassembled_module.blocks = reassembled_subblocks
                reassembled_module.skip = reassembled_skip
            elif isinstance(module, ElasticChannelHelper):
                # elastic channel helper modules are not extracted in a rebuild.
                # The active filter will be applied to each module.
                # to ensure that the length validation still works, reduce input module count by one.
                input_modules_flat_length -= 1
            else:
                logging.warn(
                    f"unknown module found during extract/rebuild '{type(module)}'. Ignoring."
                )

            if reassembled_module is not None:
                out_modules.append(reassembled_module)
            i += 1

    out_modules = flatten_module_list(out_modules)
    output_modules_flat_length = len(out_modules)
    if input_modules_flat_length != output_modules_flat_length and not quantized:
        logging.info("Reassembly changed length of module list")
    return out_modules


# return a module with the in/out channel filters of the input module applied.
# channels where the filter is false are dropped.
def apply_channel_filters(module: nn.Module) -> nn.Module:
    # first, copy the module. return a new module with channels filtered.
    module = copy.deepcopy(module)
    if not (
        isinstance(module, nn.Conv1d)
        or isinstance(module, nn.Linear)
        or isinstance(module, nn.BatchNorm1d)
    ):
        logging.error(
            f"channel filter application failed on module of invalid type: '{type(module)}'"
        )
        return module

    elastic_width_filter_input = getattr(module, "elastic_width_filter_input", None)
    elastic_width_filter_output = getattr(module, "elastic_width_filter_output", None)
    # after extracting the filters from the module, set them to None.
    # the returned module no longer requires filter application!
    setattr(module, "elastic_width_filter_input", None)
    setattr(module, "elastic_width_filter_output", None)

    if (elastic_width_filter_output is None) and (elastic_width_filter_input is None):
        # if there are no elastic filters to be applied to the module, it can be returned as is.
        return module

    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
        weight = module.weight.data.clone()
        # conv weight tensor has dimensions out_channels, in_channels, kernel_size
        # linear weight tensor has dimensions out_channels, in_channels
        # out_channel count will be length in dim 0
        out_channel_count = len(weight)
        # in_channel count will be length in second dim
        in_channel_count = len(weight[0])
        new_weight = None
        for o in range(out_channel_count):
            # if the output filter is defined and this out_channel is specified with 'False', it is dropped.
            # simply skip the iteration of this channel
            if elastic_width_filter_output is None or elastic_width_filter_output[o]:
                # if this output channel will be kept (no filter, or filter is true)
                out_channel_segment = weight[o : o + 1]
                new_out_channel = None
                for i in range(in_channel_count):
                    # segment = weight[:,i:i+1]
                    if (
                        elastic_width_filter_input is None
                        or elastic_width_filter_input[i]
                    ):
                        in_channel_segment = out_channel_segment[:, i : i + 1]
                        if new_out_channel is None:
                            # for the first in_channel being kept, simply copy over the channel
                            new_out_channel = in_channel_segment
                        else:
                            # append the input channel being kept, concatenate in dim 1 (dim 0 has length 1 and is the out_channel)
                            new_out_channel = torch.cat(
                                (new_out_channel, in_channel_segment), dim=1
                            )
                if new_out_channel is None:
                    logging.error(
                        "zero input channels were kept during channel filter application of primary module!"
                    )
                if new_weight is None:
                    # if this is the first out_channel being kept, simply copy it over
                    new_weight = new_out_channel
                else:
                    # for subsequent out_channels, cat them onto the weights in dim 0
                    new_weight = torch.cat((new_weight, new_out_channel), dim=0)
        if new_weight is None:
            logging.error(
                "zero output channels were kept during channel filter application of primary module!"
            )
        # put the new weights back into the module and return it
        module.weight.data = new_weight
        # if the module has a bias parameter, also apply the output filtering to it.
        if module.bias is not None:
            bias = module.bias.data
            new_bias = None
            for i in range(out_channel_count):
                if elastic_width_filter_output[i]:
                    if new_bias is None:
                        new_bias = bias[i : i + 1]
                    else:
                        new_bias = torch.cat((new_bias, bias[i : i + 1]), dim=0)
            logging.error(
                "zero bias channels were kept during channel filter application of primary module with bias parameter!"
            )
            module.bias.data = new_bias
        return module

    elif isinstance(module, nn.BatchNorm1d):
        elastic_filter = None
        if elastic_width_filter_output is not None:
            elastic_filter = elastic_width_filter_output
            if elastic_width_filter_input is not None:
                # this should be impossible, as it would require two channel
                # helpers with a norm but without a 'primary' module in-between
                logging.error(
                    "batchnorm1d channel filter application: a filter is specified from both sides (input, output)! defaulting to output filter."
                )
        elif elastic_width_filter_input is not None:
            elastic_filter = elastic_width_filter_input
        else:
            # this case should not be reachable
            logging.error(
                "initiated batchnorm1d channel filtering with both filters None. This is supposed to be caught earlier!"
            )
            return module
        weight = module.weight.data
        new_weight = None
        new_mean = None
        new_var = None
        channel_count = len(weight)
        for i in range(channel_count):
            if elastic_filter[i]:
                this_channel = weight[i : i + 1]
                this_mean = module.running_mean[i : i + 1]
                this_var = module.running_var[i : i + 1]
                if new_weight is None:
                    # for the first channel being kept, simply copy over
                    new_weight = this_channel
                    new_mean = this_mean
                    new_var = this_var
                else:
                    # if there are already channels being kept, concatenate this one onto the other weights
                    new_weight = torch.cat((new_weight, this_channel), dim=0)
                    new_mean = torch.cat((new_mean, this_mean), dim=0)
                    new_var = torch.cat((new_var, this_var), dim=0)
        if new_weight is None:
            logging.error(
                "zero channels were kept during channel filter application of batchnorm1d!"
            )
        # put the new weights back into the module and return it
        module.weight.data = new_weight
        module.running_var = new_var
        module.running_mean = new_mean
        return module

    else:
        # this case should not be reachable
        logging.error(
            f"initiated channel filtering for invalid module type: '{type(module)}'. This is supposed to be caught earlier!"
        )
        return module


class ModuleSet:
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

    def reassemble(
        self,
        module: nn.Conv1d,
        norm=False,
        act=False,
        norm_module: nn.BatchNorm1d = None,
        clone_conv=False,
        clone_norm=False,
    ):
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
                padding=module.padding,
            )
            new_conv.weight = module.weight
            new_conv.bias = module.bias
        modules.append(new_conv)
        if norm:
            if norm_module is None:
                raise ValueError(
                    "reassembly with norm requested, no source norm module provided"
                )
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


# TODO: Quantization
class QuantizedModuleSet1d(ModuleSet):
    conv1d = qat.Conv1d
    conv1d_norm_act = qat.ConvBnReLU1d
    conv1d_norm = qat.ConvBn1d
    conv1d_act = qat.ConvReLU1d

    def reassemble(
        self,
        module: nn.Conv1d,
        norm=False,
        act=False,
        norm_module: nn.BatchNorm1d = None,
    ):
        out_module = None
        if norm and norm_module is None:
            raise ValueError(
                "module with norm requested, no source norm module provided"
            )
        if norm and act:
            out_module = self.conv1d_norm_act(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
            )
        elif norm:
            out_module = self.conv1d_norm(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
            )
        elif act:
            out_module = self.conv1d_act(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
            )
        else:
            out_module = self.conv1d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
            )
        out_module.weight = module.weight
        out_module.bias = module.bias
        return copy.deepcopy(out_module)
