# import logging
import logging

import torch
import torch.nn as nn
import numpy as np


# Conv1d with automatic padding for the set kernel size
def conv1d_auto_padding(conv1d: nn.Conv1d):
    conv1d.padding = conv1d_get_padding(conv1d.kernel_size[0])
    return conv1d


def conv1d_get_padding(kernel_size, dilation=1):
    # check type of kernel_size
    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]

    # check type of dilation
    if isinstance(dilation, tuple):
        dilation = dilation[0]

    dil = (kernel_size - 1) * (dilation - 1)
    new_kernel_size = kernel_size + dil
    padding = new_kernel_size // 2
    return padding


# from ofa/utils/common_tools
def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


# flatten nested iterable modules, usually over a ModuleList. nn.Sequential is
# also an iterable module and a valid input.
def flatten_module_list(modules: nn.Module) -> nn.Module:
    if not hasattr(modules, "__iter__"):
        if isinstance(modules, nn.Module):
            # if the input is non-iterable and is already a module, it can be returned as a list of one element
            return nn.ModuleList([modules])

    else:
        # flatten any nested Sequential or ModuleList
        contains_nested = (isinstance(x, nn.Sequential) for x in modules) or (
            isinstance(x, nn.ModuleList) for x in modules
        )
        # repeat until the cycle no longer finds nested modules
        while contains_nested:
            # print(f"Nested? {type(modules)} {len(modules)}")
            contains_nested = False
            new_module_list = nn.ModuleList([])
            for old_item in modules:
                if hasattr(old_item, "__iter__"):
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
    if not hasattr(module_list, "__iter__"):
        if isinstance(module_list, nn.Module):
            return module_list
        raise TypeError("input is neither iterable nor module")
    if len(module_list) == 1:
        module = module_list[0]
        assert isinstance(
            module, nn.Module
        ), "Iterable single-length input does not contain module"
        return module
    else:
        return nn.Sequential(*module_list)


# recurse through any iterable (sub)structures. Attempt to call the specified
# function from any discovered objects if it is available.
# return true if any of the calls returned true
# for modules: both ModuleList and Sequential are iterable, so this should be
# able to descend into any module substructures
def call_function_from_deep_nested(input, function, type_selection: type = None):
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
    if hasattr(input, "__iter__"):
        for item in input:
            new_return_value = call_function_from_deep_nested(
                item, function, type_selection
            )
            call_return_value = call_return_value or new_return_value

    # if the object has a function to return nested modules, also check them.
    if callable(getattr(input, "get_nested_modules", None)):
        nested_modules = getattr(input, "get_nested_modules", None)()
        new_return_value = call_function_from_deep_nested(
            nested_modules, function, type_selection
        )
        call_return_value = call_return_value or new_return_value

    return call_return_value


# recurse like call_function_from_deep_nested;
# return a list of every found object of <type>
def get_instances_from_deep_nested(input, type_selection: type = None):
    results = []
    if input is None:
        return results
    if type_selection is None or isinstance(input, type_selection):
        results.append(input)
    # if the input is iterable, recursively check any nested objects
    if hasattr(input, "__iter__"):
        for item in input:
            additional_results = get_instances_from_deep_nested(item, type_selection)
            # concatenate the lists
            results += additional_results

    # if the object has a function to return nested modules, also check them.
    if callable(getattr(input, "get_nested_modules", None)):
        nested_modules = getattr(input, "get_nested_modules", None)()
        additional_results = get_instances_from_deep_nested(
            nested_modules, type_selection
        )
        results += additional_results

    return results


# copied and adapted from elasticchannelhelper.py
# compute channel priorities based on the l1 norm of the weights of whichever
# target module follows this elastic channel section
def compute_channel_priorities(module : nn.Module, kernel, channel_index : int = 0):
    channel_norms = []

    if kernel is None:
        logging.warning(
            f"Unable to compute channel priorities! Kernel is None: {kernel}"
        )
        return None
    # this will also include the elastic kernel convolutions
    # for elastic kernel convolutions, the priorities will then also be
    # computed on the base module (full kernel)
    if isinstance(module, nn.Conv1d):
        weights = kernel
        norms_per_kernel_index = torch.linalg.norm(weights, ord=1, dim=channel_index)
        channel_norms = torch.linalg.norm(norms_per_kernel_index, ord=1, dim=1)
    # the channel priorities for linears need to also be computable:
    # especially for the exit connections, a linear may follow after an elastic width
    elif isinstance(module, nn.Linear):
        weights = kernel
        channel_norms = torch.linalg.norm(weights, ord=1, dim=0)
    else:
        # the channel priorities will keep their previous / default value in
        # this case. Reduction will probably occur by channel order
        logging.warning(
            f"Unable to compute channel priorities! Unsupported target module after elastic channels: {type(module)}"
        )

    # contains the indices of the channels, sorted from channel with smallest
    # norm to channel with largest norm
    # the least important channel index is at the beginning of the list,
    # the most important channel index is at the end
    # np -> torch.argsort()
    channels_by_priority = torch.argsort(channel_norms)

    return channels_by_priority


def get_kernel_for_dpc(kernel):
    """
        At the moment uses the first kernel dimension
    """
    return kernel[:, :, 0:1]


# copied and adapted from elasticchannelhelper.py
# set the channel filter list based on the channel priorities and the reduced_target_channel count
def get_channel_filter(current_channel_size, reduced_target_channel_size, channel_priority_list):
    # get the amount of channels to be removed from the max and current channel counts
    channel_reduction_amount: int = current_channel_size - reduced_target_channel_size
    # start with an empty filter, where every channel passes through, then remove channels by priority
    channel_pass_filter = [True] * current_channel_size

    # filter the least important n channels, specified by the reduction amount
    for i in range(channel_reduction_amount):
        # priority list of channels contains channel indices from least important to most important
        # the first n channel indices specified in this list will be filtered out
        filtered_channel_index = channel_priority_list[i]
        channel_pass_filter[filtered_channel_index] = False

    return channel_pass_filter


def create_channel_filter(module: nn.Module, kernel, current_channel, reduced_target_channel_size, is_output_filter : bool = True):
    # create one channel filter
    channel_index = 1 if is_output_filter else 0
    channel_filter_priorities = compute_channel_priorities(module, kernel, channel_index)
    return get_channel_filter(current_channel, reduced_target_channel_size, channel_filter_priorities)


def filter_primary_module_weights(weights, in_channel_filter, out_channel_filter):
    # out_channel count will be length in dim 0
    out_channel_count = len(weights)
    # in_channel count will be length in second dim
    in_channel_count = len(weights[0])
    if len(in_channel_filter) != in_channel_count:
        logging.error(
            f"Unable to filter primary module weights: in_channel count {in_channel_count} does not match filter length {len(in_channel_filter)}"
        )
    if len(out_channel_filter) != out_channel_count:
        logging.error(
            f"Unable to filter primary module weights: out_channel count {out_channel_count} does not match filter length {len(out_channel_filter)}"
        )

    return (weights[out_channel_filter])[:, in_channel_filter]


def filter_single_dimensional_weights(weights, channel_filter):
    if weights is None:
        return None
    if all(channel_filter):
        return weights
    channel_count = len(weights)
    if len(channel_filter) != channel_count:
        logging.error(
            f"Unable to filter weights: channel count {channel_count} does not match filter length {len(channel_filter)}"
        )
    new_weights = None
    # channels where the filter is true are kept.
    for i in range(channel_count):
        if channel_filter[i]:
            if new_weights is None:
                new_weights = weights[i : i + 1]
            else:
                new_weights = torch.cat((new_weights, weights[i : i + 1]), dim=0)
    return new_weights


def make_parameter(t: torch.Tensor) -> nn.Parameter:
    if t is None:
        return t
    if isinstance(t, nn.Parameter):
        return t
    elif isinstance(t, torch.Tensor):
        return nn.parameter.Parameter(t)
    else:
        logging.error(f"Could not create parameter from input of type '{type(t)}'.")
        return None


# TODO not in usage, can be cleaned after evaluation
def getGroups(max_group, with_max_group_member : bool = True, addOneForNoGrouping : bool = True, divide_by : int = 2):
    tmp = [x for x in range(max_group) if x % divide_by == 0 and x != 0]
    if with_max_group_member:
        tmp.append(max_group)
    if addOneForNoGrouping and not (1 in tmp):
        tmp.append(1)
        tmp.sort(reverse=False)

    return tmp


# MR can be deleted if not needed anymore
def gather_information(module):
    """
        Collects information about the module regarding kernel adjustment with grouping
    """
    weight_adjustment_needed = is_weight_adjusting_needed(module.weight, module.in_channels, module.groups)
    target = get_target_weight(module.weight, module.in_channels, module.groups)
    if weight_adjustment_needed:
        if hasattr(module, 'id'):
            logging.debug(f"ID: {module.id}")
        logging.info(f"WARNING XKA_G ModuleName={module.__class__}  g={module.groups} ic={module.in_channels}, oc={module.out_channels}, last_g={module.last_grouping_param}")
        logging.info(f"WARNING XKA_G Weight Change is needed  {list(module.weight.shape)} target:{target}")


def adjust_weight_if_needed(module, kernel=None, groups=None):
    """
    Adjust the weight if the adjustment is needded. This means, if the kernel does not have the size of
    (out_channel, in_channel / group, kernel).

    :param: kernel the kernel that should be checked and adjusted if needed. If None module.weight.data will be used
    :param: grouping value of the conv, if None module.groups will be used
    :param: module the conv

    :throws: RuntimeError if there is no last_grouping_param for comporing current group value to past group value

    returns (kernel, is adjusted) (adjusted if needed) otherwise throws a RuntimeError
    """
    if kernel is None:
        kernel = module.weigth.data
    if groups is None:
        groups = module.groups

    if not hasattr(module, 'last_grouping_param'):
        raise RuntimeError

    in_channels = kernel.size(1)

    is_adjusted = False

    grouping_changed = groups != module.last_grouping_param
    logging.debug(f"Shape:{kernel.shape} Groups:{groups} Group_First: {module.last_grouping_param} groups_changed:{grouping_changed} ic={module.in_channels}, oc={module.out_channels}")
    if grouping_changed and groups > 1:
        weight_adjustment_needed = is_weight_adjusting_needed(kernel, in_channels, groups)
        if weight_adjustment_needed:
            is_adjusted = True
            logging.debug(f"NOW Shape:{kernel.shape} Groups:{groups} Group_First: {module.last_grouping_param} groups_changed:{grouping_changed} ic={module.in_channels}, oc={module.out_channels}")
            kernel = adjust_weights_for_grouping(kernel, groups)
        else:
            target = get_target_weight(kernel, in_channels, groups)
            if hasattr(module, 'id'):
                logging.debug(f"ID: {module.id}")
            logging.debug(f"XKA ModuleName={module.__class__}  g={groups} ic={module.in_channels}, oc={module.out_channels}")
            logging.debug(f"XKA Grouping changed BUT no weight change is needed - hurray! {list(kernel.shape)} target:{target}")

    return (kernel, is_adjusted)


def is_weight_adjusting_needed(weights, input_channels, groups):
    """
        Checks if a weight adjustment is needed
        Requirement: weight.shape[1] must be input_channels/groups
        true: weight adjustment is needed
        :param: weights - the weights that needs to be checked
        :param: input_channels - Input Channels of the Convolution Module
        :param: groups - Grouping Param of the Convolution Module
    """
    current_weight_dimension = weights.shape[1]
    target_weight_dimension = input_channels // groups
    return target_weight_dimension != current_weight_dimension


def get_target_weight(weights, input_channels, groups):
    """
        Gives the targeted weight shape (out_channel, in_channel // groups, kernel)
        :param: weights - the weights that needs to be checked
        :param: input_channels - Input Channels of the Convolution Module
        :param: groups - Grouping Param of the Convolution Module
    """
    target_shape = list(weights.shape)
    target_shape[1] = input_channels // groups
    return target_shape


def prepare_kernel_for_depthwise_separable_convolution(kernel, bias, in_channel_count, in_channel_filter, out_channel_filter):
    # if in_channel_count != len(in_channel_filter):
    #     logging.warning(f"input channel filter has not the same length as the given input channel count. \
    #     filter:{len(in_channel_filter)} size: {in_channel_count}")
    #     return None TODO: need sum not

    # outchannel is adapted
    new_kernel = filter_primary_module_weights(kernel, in_channel_filter, out_channel_filter)
    # grouping = in_channel_count
    new_kernel = adjust_weights_for_grouping(new_kernel, in_channel_count)

    if bias is None:
        return new_kernel, None
    else:
        new_bias = filter_single_dimensional_weights(
            bias, out_channel_filter
        )
    return new_kernel, new_bias


def prepare_kernel_for_pointwise_convolution(kernel, bias, in_channel_filter, out_channel_filter, grouping):
    # outchannel is adapted
    new_kernel = filter_primary_module_weights(kernel, in_channel_filter, out_channel_filter)
    # use 1x1 kernel
    new_kernel = get_kernel_for_dpc(kernel)
    # grouping = in_channel_count
    new_kernel = adjust_weights_for_grouping(new_kernel, grouping)

    if bias is None:
        return new_kernel, None
    else:
        new_bias = filter_single_dimensional_weights(
            bias, out_channel_filter
        )
    return new_kernel, new_bias


def adjust_weights_for_grouping(weights, input_divided_by):
    """
        Adjusts the Weights for the Forward of the Convulution
        Shape(outchannels, inchannels / group, kW)
        weight – filters of shape (out_channels , in_channels / groups , kW)
        input_divided_by
    """
    channels_per_group = weights.shape[1] // input_divided_by

    splitted_weights = torch.tensor_split(weights, input_divided_by)
    result_weights = []

    # for current_group in range(groups):
    for current_group, current_weight in enumerate(splitted_weights):
        input_start = current_group * channels_per_group
        input_end = input_start + channels_per_group
        current_result_weight = current_weight[:, input_start:input_end, :]
        result_weights.append(current_result_weight)

    full_kernel = torch.concat(result_weights)

    return full_kernel
