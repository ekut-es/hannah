import copy
import torch.nn as nn
import numpy as np
import logging
# import torch
# from ..utils import ConfigType, SerializableModule
from ..factory import qat as qat


# Conv1d with automatic padding for the set kernel size
def conv1d_auto_padding(conv1d: nn.Conv1d):
    conv1d.padding = conv1d.kernel_size[0] // 2
    return conv1d


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


# TODO: properly implement stride on major blocks
def create(name: str, labels: int, input_shape, conv=[], min_depth: int = 1, norm_order=None, steps_without_sampling=1):
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
        steps_without_sampling=steps_without_sampling
    )

    return model


# build a sequence from a list of minor block configurations
def create_minor_block_sequence(blocks, in_channels, stride=1, norm_order=None):
    next_in_channels = in_channels
    minor_block_sequence = nn.ModuleList([])
    norm_before_act = norm_order.norm_before_act
    norm_after_act = norm_order.norm_after_act
    is_first_minor_block = True
    for block_config in blocks:
        if block_config.target == "conv1d":
            out_channels = block_config.out_channels
            # create a minor block, potentially with activation and norm
            minor_block_internal_sequence = nn.ModuleList([])
            new_minor_block = conv1d_auto_padding(nn.Conv1d(
                    kernel_size=block_config.kernel_size,
                    in_channels=next_in_channels,
                    out_channels=out_channels
            ))
            # set stride on the first minor block in the sequence
            if is_first_minor_block:
                new_minor_block.stride = stride
                is_first_minor_block = False
            minor_block_internal_sequence.append(new_minor_block)
            # batch norm will be added before and/or after activation depending on the configuration
            if block_config.norm and norm_before_act:
                minor_block_internal_sequence.append(nn.BatchNorm1d(out_channels))
            if block_config.act:
                # add relu activation if act is set
                minor_block_internal_sequence.append(nn.ReLU())
            if block_config.norm and norm_after_act:
                minor_block_internal_sequence.append(nn.BatchNorm1d(out_channels))

            minor_block_sequence.append(nn.Sequential(*minor_block_internal_sequence))
            # the input channel count of the next minor block is the output channel count of the previous block
            next_in_channels = out_channels
        # if an unknown target is selected for a minor block, throw an exception.
        else:
            raise Exception(f"Undefined target selected in minor block sequence: {block_config.target}")

    return nn.Sequential(*minor_block_sequence)


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
    def __init__(self, conv_layers: nn.ModuleList([]), max_depth: int, labels: int, pool_kernel: int, flatten_dims: int, out_channels: int, min_depth: int = 1, block_config=[], steps_without_sampling: int = 1):
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
        self.current_step = 0
        self.last_input = None
        self.pool = nn.AvgPool1d(pool_kernel)
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

    # filter from DynamicConv2d
    def get_active_filter(self, out_channel, in_channel):
        # out_channels, in_channels/groups, kernel_size[0], kernel_size[1]
        return self.conv.weight[:out_channel, :in_channel, :]

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

    def get_elastic_depth_output(self, target_depth=None):
        if target_depth is None:
            target_depth = self.max_depth
        if self.last_input is None:
            return None
        submodel = self.extract_elastic_depth_sequence(target_depth)
        # print(submodel)
        # print(type(submodel))
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

    def should_subsample(self, verify_step=False):
        # Shortcut for testing: set to True to also verify loss of equivalent extracted model
        if verify_step:
            return False
        return self.current_step > self.steps_without_sampling

    # reset elastic values to their default (max) values
    def reset_active_elastic_values(self):
        self.active_depth = self.max_depth

    # set the output channel count value based on the current active depth
    def update_output_channel_count(self):
        # the new out channel count is given by the last minor block of the last active major block
        self.out_channels = self.block_config[:self.active_depth][-1].blocks[-1].out_channels

    def get_output_linear_layer(self, target_depth):
        return self.linears[target_depth-self.min_depth]


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

        """
        modules = nn.ModuleList([])
        for i, item in enumerate(block):
            rebuild_output = rebuild_extracted_block(item, quantized)
            if rebuild_output is not None:
                for output_item in rebuild_output:
                    modules.append(output_item)
    # return module_set.assemble(weights, norm, act)
        """
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
            out_module = (module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding)
        out_module.weight = module.weight
        out_module.bias = module.bias
        return copy.deepcopy(out_module)
