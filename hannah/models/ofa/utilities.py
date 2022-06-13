# import logging
import logging
import torch
import torch.nn as nn


# Conv1d with automatic padding for the set kernel size
def conv1d_auto_padding(conv1d: nn.Conv1d):
    conv1d.padding = conv1d_get_padding(conv1d.kernel_size[0])
    return conv1d


def conv1d_get_padding(kernel_size, dilation=1):
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


def filter_primary_module_weights(weights, in_channel_filter, out_channel_filter):
    # HIER KNALLTS
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


def getGroups(max_group, with_max_group_member : bool = True, addOneForNoGrouping : bool = True, divide_by : int = 2):
    tmp = [x for x in range(max_group) if x % divide_by == 0 and x != 0]
    if with_max_group_member:
        tmp.append(max_group)
    if addOneForNoGrouping and not (1 in tmp):
        tmp.append(1)
        tmp.sort(reverse=False)

    return tmp

def adjust_weights_for_grouping(weights, input_divided_by=2):
    """
        Adjusts the Weights for the Forward of the Convulution
        Shape(outchannels, inchannels / group, kW)
        weight – filters of shape (out_channels , in_channels / groups , kW)
        input_divided_by
    """


    # if(not self.training):
    #     logging.info(f"Validation Step, Weight Shape is {weights.shape}")
    #     logging.info(f"New Weight Shape is {full_kernel.shape}")

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

    # print(full_kernel.shape)
    return full_kernel


def restore_shape_weights(weights, input_was_divided_by=2):
    """
    """

    # if(not self.training):
    #     logging.info(f"Validation Step, Weight Shape is {weights.shape}")
    #     logging.info(f"New Weight Shape is {full_kernel.shape}")

    channels_per_group = weights.shape[1] * input_was_divided_by

    splitted_weights = torch.tensor_split(weights, input_was_divided_by)
    result_weights = []

    # for current_group in range(groups):
    for current_group, current_weight in enumerate(splitted_weights):
        input_start = current_group * channels_per_group
        input_end = input_start + channels_per_group
        current_result_weight = current_weight[:, input_start:input_end, :]
        result_weights.append(current_result_weight)

    full_kernel = torch.concat(result_weights)

    # print(full_kernel.shape)
    return full_kernel
