import copy
from typing import List, Tuple
import torch.nn as nn

# import torch.nn.functional as nnf
import numpy as np
import logging

# import torch

# from ..utils import ConfigType, SerializableModule
from omegaconf import ListConfig
from hydra.utils import instantiate

from ..factory import qat as qat
from .submodules.elasticchannelhelper import ElasticChannelHelper, SequenceDiscovery
from .submodules.elastickernelconv import (
    ElasticConv1d,
    ElasticConvBn1d,
    ElasticConvBnReLu1d,
    ElasticQuantConvBn1d,
)
from .submodules.resblock import ResBlock1d, ResBlockBase
from .submodules.elasticwidthmodules import (
    ElasticPermissiveReLU,
    ElasticWidthBatchnorm1d,
    ElasticWidthLinear,
)

# from .submodules.sequencediscovery import SequenceDiscovery
from .utilities import (
    flatten_module_list,
    get_instances_from_deep_nested,
    # set_basic_weight_grad,
    module_list_to_module,
    # conv1d_get_padding,
    call_function_from_deep_nested,
    # set_weight_maybe_bias_grad,
)


def create(
    name: str,
    labels: int,
    input_shape,
    conv=[],
    min_depth: int = 1,
    norm_before_act=True,
    skew_sampling_distribution: bool = False,
    dropout: int = 0.5,
    validate_on_extracted=True,
    qconfig=None,
) -> nn.Module:
    # if no orders for the norm operator are specified, fall back to default
    default_qconfig = instantiate(qconfig) if qconfig else None
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

    for block_config in conv:
        if block_config.target == "forward":
            major_block = create_forward_block(
                blocks=block_config.blocks,
                in_channels=next_in_channels,
                stride=block_config.stride,
                norm_before_act=norm_before_act,
                qconfig=default_qconfig
                # sources=previous_sources,
            )

        elif block_config.target == "residual1d":
            major_block = create_residual_block_1d(
                blocks=block_config.blocks,
                in_channels=next_in_channels,
                stride=block_config.stride,
                norm_before_act=norm_before_act,
                qconfig=default_qconfig
                # sources=previous_sources,
            )

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
        skew_sampling_distribution=skew_sampling_distribution,
        dropout=dropout,
        validate_on_extracted=validate_on_extracted,
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

    model.perform_sequence_discovery()

    return model


# build a sequence from a list of minor block configurations
def create_minor_block_sequence(
    blocks,
    in_channels,
    stride=1,
    norm_before_act=True,
    qconfig=None,
    # sources: List[nn.Module] = [nn.ModuleList([])],
):
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
        minor_block, next_in_channels = create_minor_block(
            block_config=block_config,
            in_channels=next_in_channels,
            stride=next_stride,
            norm_before_act=norm_before_act,
            qconfig=qconfig,
            # sources=sources,
        )

        minor_block_sequence.append(minor_block)

    # return module_list_to_module(minor_block_sequence), elastic_helper
    return module_list_to_module(minor_block_sequence)


# build a single minor block from its config. return the number of output channels with the block
def create_minor_block(
    block_config,
    in_channels: int,
    stride: int = 1,
    norm_before_act=True,
    sources: List[nn.ModuleList] = [nn.ModuleList([])],
    qconfig=None,
) -> Tuple[nn.Module, int]:
    new_block = None
    # the output channel count is usually stored in block_config.out_channels
    # use it as the default value if available, otherwise it must be set by the specific code handling the target type
    new_block_out_channels = getattr(block_config, "out_channels", 1)

    if "conv1d" in block_config.target:
        out_channels = block_config.out_channels
        if not isinstance(out_channels, ListConfig):
            out_channels = [out_channels]
        out_channels.sort(reverse=True)
        # the maximum available width is the initial output channel count
        out_channels_full = out_channels[0]

        kernel_sizes = block_config.kernel_sizes
        if not isinstance(kernel_sizes, ListConfig):
            kernel_sizes = [kernel_sizes]

        minor_block_internal_sequence = nn.ModuleList([])
        norm = block_config.get("norm", False)
        act = block_config.get("act", False)

        if not norm and not act:
            new_minor_block = ElasticConv1d(
                kernel_sizes=kernel_sizes,
                in_channels=in_channels,
                out_channels=out_channels_full,
                stride=stride,
                # padding=conv1d_get_padding(block_config.kernel_size)  # elastic kernel conv will autoset padding
            )
        elif norm and not act:
            new_minor_block = ElasticQuantConvBn1d(
                kernel_sizes=kernel_sizes,
                in_channels=in_channels,
                out_channels=out_channels_full,
                stride=stride,
                qconfig=qconfig,
                # padding=conv1d_get_padding(block_config.kernel_size)  # elastic kernel conv will autoset padding
            )
        elif norm and act:
            new_minor_block = ElasticConvBnReLu1d(
                kernel_sizes=kernel_sizes,
                in_channels=in_channels,
                out_channels=out_channels_full,
                stride=stride,
                # padding=conv1d_get_padding(block_config.kernel_size)  # elastic kernel conv will autoset padding
            )
        else:
            raise Exception(
                f"Undefined target selected in minor block sequence: {block_config.target}"
            )

        minor_block_internal_sequence.append(new_minor_block)

        new_block = module_list_to_module(
            flatten_module_list(minor_block_internal_sequence)
        )
        # if multiple output channel widths are specified (elastic width), add an elastic width helper module
        if len(out_channels) > 1:
            # the sources of the elastic channel helper module are the previous conv, and its potential norm/act
            helper_module = ElasticChannelHelper(out_channels)
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
        new_block = ElasticChannelHelper(out_channels_list)

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
    norm: bool, act: bool, channels: int, norm_before_act=None
) -> nn.Module:
    # batch norm will be added before and/or after activation depending on the configuration
    # fallback default is one norm before act, if no order is specified.

    # if no norm or activation is requested, simply return None
    # going through the steps below and returning an empty module list would also be fine
    if not norm and not act:
        return None

    norm_act_sequence = nn.ModuleList([])
    # create the norm module only if required. its reference will be passed back.
    new_norm = None
    if norm:
        new_norm = ElasticWidthBatchnorm1d(channels)
    # new_act = nn.ReLU()
    new_act = ElasticPermissiveReLU()
    if norm and norm_before_act:
        norm_act_sequence.append(new_norm)
    if act:
        # add relu activation if act is set
        norm_act_sequence.append(new_act)
    if norm and not norm_before_act:
        norm_act_sequence.append(new_norm)

    return module_list_to_module(norm_act_sequence)


# build a basic forward major block
def create_forward_block(
    blocks,
    in_channels,
    stride=1,
    norm_before_act=None,
    qconfig=None
    # sources: List[nn.ModuleList] = [nn.ModuleList([])],
):
    return create_minor_block_sequence(
        blocks,
        in_channels,
        stride=stride,
        norm_before_act=norm_before_act,
        qconfig=qconfig,
    )


# build a residual major block
def create_residual_block_1d(
    blocks,
    in_channels,
    stride=1,
    norm_before_act=None,
    qconfig=None
    # sources: List[nn.ModuleList] = [nn.ModuleList([])],
) -> ResBlock1d:
    minor_blocks = create_minor_block_sequence(
        blocks,
        in_channels,
        stride=stride,
        norm_before_act=norm_before_act,
        qconfig=qconfig,
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
        norm_before_act=norm_before_act,
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
        skew_sampling_distribution=False,
        dropout=0.5,
        validate_on_extracted=False,
    ):
        super().__init__()
        self.validate_on_extracted = validate_on_extracted
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
        self.sampling_max_width_step = 0
        self.eval_mode = False
        self.last_input = None
        self.skew_sampling_distribution = skew_sampling_distribution
        self.validation_model = None

        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(flatten_dims)

        # one linear exit layer for each possible depth level
        self.linears = nn.ModuleList([])
        # for every possible depth level (from min_depth to including max_depth)
        for i in range(self.min_depth, self.max_depth + 1):
            self.active_depth = i
            self.update_output_channel_count()
            # create the linear output layer for this depth
            new_output_linear = ElasticWidthLinear(self.out_channels, self.labels)
            self.linears.append(new_output_linear)

        # should now be redundant, as the loop will exit with the active depth being max_depth
        self.active_depth = self.max_depth
        # ofa step counts will be set by the create function.
        self.ofa_steps_kernel = 1
        self.ofa_steps_depth = 1
        self.ofa_steps_width = 1
        # create a list of every elastic kernel conv, for sampling
        all_elastic_kernel_convs = get_instances_from_deep_nested(
            input=self.conv_layers,
            type_selection=(ElasticConv1d, ElasticConvBn1d, ElasticConvBnReLu1d),
        )
        self.elastic_kernel_convs = []
        for item in all_elastic_kernel_convs:
            if item.get_available_kernel_steps() > 1:
                # ignore convs with only one available kernel size, they do not need to be stored
                self.elastic_kernel_convs.append(item)
        logging.info(
            f"OFA model accumulated {len(self.elastic_kernel_convs)} elastic kernel convolutions for sampling."
        )

        # create a list of every elastic width helper, for sampling
        self.elastic_channel_helpers = get_instances_from_deep_nested(
            input=self.conv_layers, type_selection=ElasticChannelHelper
        )
        logging.info(
            f"OFA model accumulated {len(self.elastic_channel_helpers)} elastic width connections for sampling."
        )

    def forward(self, x):
        self.last_input = x
        self.current_step = self.current_step + 1

        # in eval mode, run the forward on the extracted validation model.
        if self.eval_mode and self.validate_on_extracted:
            if self.validation_model is None:
                logging.warn(
                    "forward in validation mode called without building validation model!"
                )
                self.build_validation_model()
            return self.validation_model.forward(x)

        # if the network is currently being evaluated, don't sample a subnetwork!
        if (
            self.sampling_max_depth_step > 0
            or self.sampling_max_kernel_step > 0
            or self.sampling_max_width_step > 0
        ) and not self.eval_mode:
            self.sample_subnetwork()
        for layer in self.conv_layers[: self.active_depth]:
            x = layer(x)

        result = x
        result = self.pool(result)
        result = self.flatten(result)
        result = self.dropout(result)
        result = self.get_output_linear_layer(self.active_depth)(result)

        return result

    def perform_sequence_discovery(self):
        logging.info("Performing model sequence discovery.")
        # start with a new, empty sequence discovery
        sequence_discovery = SequenceDiscovery(is_accumulating_sources=True)
        per_layer_output_discoveries = []
        for layer in self.conv_layers:
            resulting_discovery = layer(sequence_discovery)
            # for each layer, store a split discovery for the output linear at that layer.
            # THESE MUST BE APPLIED AFTER THE FULL MODULE DISCOVERY IS COMPLETED
            # to ensure that the primary targets are set correctly.
            per_layer_output_discoveries.append(resulting_discovery.split())
            sequence_discovery = resulting_discovery

        # after the layers are processed, pass the relevant SequenceDiscovery to each output linear
        for i in range(self.min_depth, self.max_depth + 1):
            # range goes from min depth to including the max depth
            output_linear = self.get_output_linear_layer(i)
            sequence_discovery = per_layer_output_discoveries[i - 1]
            output_linear.forward(sequence_discovery)
            # the resulting output sequence discovery is dropped. no module trails the output linear.

    # pick a random subnetwork, return the settings used
    def sample_subnetwork(self):
        state = {"depth_step": 0, "kernel_steps": [], "width_steps": []}
        # new_depth_step = np.random.randint(self.sampling_max_depth_step+1)
        new_depth_step = self.get_random_step(self.sampling_max_depth_step + 1)
        self.active_depth = self.max_depth - new_depth_step
        state["depth_step"] = new_depth_step

        for conv in self.elastic_kernel_convs:
            # pick an available kernel index for every elastic kernel conv, independently.
            max_available_sampling_step = min(
                self.sampling_max_kernel_step + 1, conv.get_available_kernel_steps()
            )
            new_kernel_step = self.get_random_step(max_available_sampling_step)
            conv.pick_kernel_index(new_kernel_step)
            state["kernel_steps"].append(new_kernel_step)

        """for helper in self.elastic_channel_helpers:
            # pick an available width step for every elastic channel helper, independently.
            max_available_sampling_step = min(
                self.sampling_max_width_step + 1, helper.get_available_width_steps()
            )
            new_width_step = self.get_random_step(max_available_sampling_step)
            helper.set_channel_step(new_width_step)
            state["width_steps"].append(new_width_step)
        """
        return state

    # get a step, with distribution biased towards taking less steps, if skew distribution is enabled.
    # currently a sort-of pseudo-geometric distribution, may be replaced with better RNG
    def get_random_step(self, upper_bound: int) -> int:
        if upper_bound <= 0:
            logging.warn("requested impossible random step <= 0. defaulting to 0.")
            return 0
        if (not self.skew_sampling_distribution) or self.eval_mode:
            # during random submodel evaluation, use uniform distribution
            return np.random.randint(upper_bound)
        else:
            acc = 0
            while np.random.randint(2) and acc < upper_bound:
                # continue incrementing with a 1/2 chance per additional increment
                acc += 1
            if acc == upper_bound:
                # if the bound was reached, go back below the bound.
                # due to this, the distribution of probability toward the last element
                # is not consistent with the distribution gradient across other elements
                acc -= 1
            return acc

    # return max available step values
    def get_max_submodel_steps(self):
        max_depth_step = self.sampling_max_depth_step
        kernel_steps = []
        width_steps = []
        for conv in self.elastic_kernel_convs:
            kernel_steps.append(conv.get_available_kernel_steps())
        for helper in self.elastic_channel_helpers:
            width_steps.append(helper.get_available_width_steps())
        state = {
            "depth_step": max_depth_step,
            "kernel_steps": kernel_steps,
            "width_steps": width_steps,
        }
        return state

    # accept a state dict like the one returned in get_max_submodel_steps, return extracted submodel.
    # also sets main model state to this submodel.
    def get_submodel(self, state: dict):
        if not self.set_submodel(state):
            return None
        else:
            return self.extract_elastic_depth_sequence(self.active_depth)

    # accept a state dict like the one returned in get_max_submodel_steps, sets model state.
    def set_submodel(self, state: dict):
        try:
            depth_step = state["depth_step"]
            kernel_steps = state["kernel_steps"]
            width_steps = state["width_steps"]
        except KeyError:
            logging.error(
                "Invalid state dict passed to get_submodel! Keys should be 'depth_step', 'kernel_steps', 'width_steps'!"
            )
            return False
        if len(kernel_steps) != len(self.elastic_kernel_convs):
            print(
                f"State dict provides invalid amount of kernel steps: model has {len(self.elastic_kernel_convs)}, {len(kernel_steps)} provided."
            )
            return False
        if len(width_steps) != len(self.elastic_channel_helpers):
            print(
                f"State dict provides invalid amount of width steps: model has {len(self.elastic_channel_helpers)}, {len(width_steps)} provided."
            )
            return False

        self.active_depth = self.max_depth - depth_step
        for i in range(len(kernel_steps)):
            self.elastic_kernel_convs[i].pick_kernel_index(kernel_steps[i])
        for i in range(len(width_steps)):
            self.elastic_channel_helpers[i].set_channel_step(width_steps[i])

        return True

    def build_validation_model(self):
        self.validation_model = self.extract_elastic_depth_sequence(self.active_depth)

    def get_validation_model_weight_count(self):
        if self.validation_model is None:
            return 0
        else:
            # create a dict of the pointer of each parameter to the item count within that parameter
            # using a dict with pointers as keys ensures that no parameter is counted twice
            parameter_pointers_dict = dict(
                (p.data_ptr(), p.numel()) for p in self.validation_model.parameters()
            )
            # sum up the values of each dict item, yielding the total element count across params
            return sum(parameter_pointers_dict.values())

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
        extracted_module_list.append(self.dropout)
        output_linear = self.get_output_linear_layer(target_depth)
        if isinstance(output_linear, ElasticWidthLinear):
            output_linear = output_linear.assemble_basic_linear()
        extracted_module_list.append(output_linear)
        # extracted_module_list = flatten_module_list(extracted_module_list)
        # return copy.deepcopy(module_list_to_module(extracted_module_list))
        return copy.deepcopy(nn.Sequential(*extracted_module_list))

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
        return call_function_from_deep_nested(
            input=self.conv_layers,
            function="step_down_channels",
            type_selection=ElasticChannelHelper,
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

        # the code below this is probably no longer doing anything, TBD.
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

    # return the linear layer which processes the output for the current elastic depth
    def get_output_linear_layer(self, target_depth):
        return self.linears[target_depth - self.min_depth]

    # step all elastic kernels within the model down by one, if possible
    def step_down_all_kernels(self):
        return call_function_from_deep_nested(
            input=self.conv_layers,
            function="step_down_kernel_size",
            type_selection=ElasticConv1d,
        )

    # reset all kernel sizes to their max value
    def reset_all_kernel_sizes(self):
        return call_function_from_deep_nested(
            input=self.conv_layers,
            function="reset_kernel_size",
            type_selection=ElasticConv1d,
        )

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

    # reset all kernel sizes to their max value
    def reset_all_widths(self):
        return call_function_from_deep_nested(
            input=self.conv_layers,
            function="reset_channel_step",
            type_selection=ElasticChannelHelper,
        )

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
                f"excessive OFA depth stepping! Attempting to add a depth step when max ({self.ofa_steps_depth}) already reached"
            )

    def progressive_shrinking_compute_channel_priorities(self):
        call_function_from_deep_nested(
            input=self.conv_layers,
            function="compute_channel_priorities",
            type_selection=ElasticChannelHelper,
        )

    def progressive_shrinking_add_width(self):
        self.sampling_max_width_step += 1
        if self.sampling_max_width_step >= self.ofa_steps_width:
            self.sampling_max_width_step -= 1
            logging.warn(
                f"excessive OFA depth stepping! Attempting to add a width step when max ({self.ofa_steps_width}) already reached"
            )

    def progressive_shrinking_disable_sampling(self):
        self.sampling_max_kernel_step = 0
        self.sampling_max_depth_step = 0
        self.sampling_max_width_step = 0


def is_elastic_module(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            ElasticConv1d,
            ElasticQuantConvBn1d,
            ElasticWidthBatchnorm1d,
            ElasticWidthLinear,
            ElasticPermissiveReLU,
        ),
    )


def assemble_basic_from_elastic_module(module: nn.Module) -> nn.Module:
    if isinstance(
        module,
        (ElasticConv1d, ElasticConvBn1d, ElasticConvBnReLu1d, ElasticQuantConvBn1d),
    ):
        return module.assemble_basic_conv1d()
    elif isinstance(module, ElasticWidthBatchnorm1d):
        return module.assemble_basic_batchnorm1d()
    elif isinstance(module, ElasticWidthLinear):
        return module.assemble_basic_linear()
    elif isinstance(module, ElasticPermissiveReLU):
        return nn.ReLU()
    else:
        logging.info(
            f"requested basic module for non-elastic source module: {type(module)}"
        )
        return module


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

    if isinstance(blocks, (nn.Sequential, nn.ModuleList)):
        modules = nn.ModuleList([])
        for item in blocks:
            modules.append(item)

        modules = flatten_module_list(modules)

        input_modules_flat_length = len(modules)

        # if the module is an elastic module, it is replaced by an equivalent basic module for its current state
        for i in range(len(modules)):
            module = modules[i]
            if is_elastic_module(module):
                modules[i] = assemble_basic_from_elastic_module(module)

        i = 0
        while i in range(len(modules)):
            module = modules[i]
            reassembled_module = None

            if isinstance(module, nn.Conv1d):
                # apply channel filters to the conv, if present.
                if i + 1 in range(len(modules)) and isinstance(
                    modules[i + 1], nn.BatchNorm1d
                ):
                    # apply channel filters to the norm, if present.
                    norm_module = modules[i + 1]
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
                reassembled_module = module
                # for standalone batchnorms, apply any channel filters, if present.
                # if module_set.norm1d is not None:
                #     pass the channel count on to the new norm type
                #     reassembled_module = module_set.norm1d(module.num_features)
                #     reassembled_module.weight = module.weight
                # else:
                #     logging.error(
                #         "Skipping stand-alone norm in reassembly: not available in the selected module set"
                #     )
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
                norm = module.norm
                act = module.act
                if is_elastic_module(norm):
                    norm = assemble_basic_from_elastic_module(norm)
                if is_elastic_module(act):
                    act = assemble_basic_from_elastic_module(act)
                reassembled_module.norm_before_act = module.norm_before_act
                reassembled_module.do_act = module.do_act
                reassembled_module.do_norm = module.do_norm
                reassembled_module.norm = norm
                reassembled_module.act = act

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
