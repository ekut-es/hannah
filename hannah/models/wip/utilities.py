import torch.nn as nn


# Conv1d with automatic padding for the set kernel size
def conv1d_auto_padding(conv1d: nn.Conv1d):
    conv1d.padding = conv1d_get_padding(conv1d.kernel_size[0])
    return conv1d


def conv1d_get_padding(kernel_size):
    padding = kernel_size // 2
    return padding


# from ofa/utils/common_tools
def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


# flatten nested iterable modules, usually over a ModuleList. nn.Sequential is also an iterable module and a valid input.
def flatten_module_list(modules: nn.Module) -> nn.Module:
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
