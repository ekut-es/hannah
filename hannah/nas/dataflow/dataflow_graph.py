from typing import Iterable
from copy import copy
import re
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.optional_op import OptionalOp
from hannah.nas.dataflow.tensor_type import TensorType


class DataFlowGraph:
    def __init__(self, inputs, output, name: str = "dataflow") -> None:
        self.inputs = inputs
        self.output = output
        self.name = name
        self.id = name
        insert_scope_to_id(self, {}, {}, [], {}, {})
        names = []
        get_names(self, names)
        name_map = get_correct_hierarchy_map(names)
        fix_hierarchy(self, name_map)

        # self.leaf_nodes = collect_leaf_nodes(self.output)

    def output_tensor(self):
        # TODO: what happens with multiple outputs?
        return self.output[0].output_tensor()

    def dfg_line_representation(self, key, indent):
        return '\t'*indent + key + ':'

    def get_string(self):
        lines = []
        hierarchy_dict = {}
        input_names = {}
        tensors = {}
        get_hierarchical_dict(self,
                              hierarchy_dict=hierarchy_dict,
                              current_scope=[],
                              inputs={},
                              scopes={},
                              input_names=input_names,
                              nested_scopes={},
                              scope_counters={},
                              tensors=tensors)

        for key, val in reversed(tensors.items()):
            lines.append('%{{{}}} = {}'.format(val, key))

        self.get_str_lines_from_dict(hierarchy_dict, lines, 0, input_names)
        mapping = get_correct_hierarchy_map(lines)
        lines = mapping.values()
        return "\n".join(lines)

    def get_str_lines_from_dict(self, container, lines, indent, input_names):
        for key, val in reversed(container.items()):
            if isinstance(val, dict):
                lines.append(self.dfg_line_representation(key, indent))
                self.get_str_lines_from_dict(val, lines, indent+1, input_names)
            elif isinstance(val, list):
                lines.append('\t'*indent + '%{{{}}} = {}('.format(input_names[key], key) +
                             ', '.join(['%{{{}}}' for _ in range(len(val))]).format(*[input_names[x] for x in val]) +
                             ')')

    def __str__(self):
        return self.get_string()

    def __repr__(self) -> str:
        return str(self)


def dataflow(func):
    def wrapper_func(*args, **kwargs):
        name = func.__name__
        inputs = args
        output = func(*args, **kwargs)

        if isinstance(output, Iterable):
            output = tuple(output)
        else:
            output = (output,)
        dfg = DataFlowGraph(inputs=inputs, output=output, name=name)
        return dfg

    return wrapper_func


def expose_dataflow_outputs(args: tuple):
    exposed_args = []
    for arg in args:
        if isinstance(arg, DataFlowGraph):
            exposed_args.extend(arg.outputs)
        else:
            exposed_args.append(arg)
    return tuple(exposed_args)


def collect_leaf_nodes(g):
    def _propagate(x, leafs):
        if isinstance(x, Iterable):
            for i in x:
                leafs = _propagate(i, leafs)
            return leafs
        elif isinstance(x, DataFlowGraph):
            leafs.extend(x.inputs)
            return leafs
        elif isinstance(x, OpType):
            for inp in x.operands:
                leafs = _propagate(inp, leafs)
            return leafs
        elif isinstance(x, OptionalOp):
            leafs = _propagate(x.op, leafs)
            return leafs
        elif isinstance(x, TensorType):
            return leafs + [x]

    leafs = _propagate(g, [])
    return leafs


def register_scope(scope, scope_counters):
    if scope not in scope_counters:
        scope_counters[scope] = 0
    else:
        scope_counters[scope] += 1
    return scope + ".{{{}}}".format(scope_counters[scope])


def scope_nester(scope, current_scope, nesting):
    nesting[".".join(current_scope)] = scope
    return nesting


def reset_nested_counters(ended_scope, nesting, scope_counter):
    if ended_scope in nesting:
        base_scope_str = nesting[ended_scope].split(".")[0]
        scope_counter.pop(base_scope_str)


def insert_scope_to_id(x, inputs, scopes, current_scope, scope_counters, nested_scopes):
    if isinstance(x, DataFlowGraph):
        if x in inputs:
            for i in inputs[x]:
                current_scope.remove(scopes[i])
                reset_nested_counters(scopes[i], nested_scopes, scope_counters)
        scopes[x] = register_scope(x.name, scope_counters)
        x.id = ".".join(current_scope) + ".{}".format(scopes[x]) if current_scope else scopes[x]
        inp = x.inputs[0]
        if inp in inputs:
            inputs[inp].append(x)
        else:
            inputs[inp] = [x]
        nested_scopes = scope_nester(scopes[x], current_scope, nested_scopes)
        current_scope += [scopes[x]]
        # TODO: Handle multiple outputs
        insert_scope_to_id(x.output[0], inputs, scopes, current_scope, scope_counters, nested_scopes)
    elif isinstance(x, OpType):
        x.id = ".".join(current_scope) + ".{}".format(x.name)

        for o in x.operands:
            cs = copy(current_scope)
            insert_scope_to_id(o, inputs, scopes, cs, scope_counters, nested_scopes)

        if x in inputs:
            for i in inputs[x]:
                current_scope.remove(scopes[i])
                reset_nested_counters(scopes[i], nested_scopes, scope_counters)

    elif isinstance(x, TensorType):
        x.id = ".".join(current_scope) + ".{}".format(x.name)
        if x in inputs:
            for i in inputs[x]:
                current_scope.remove(scopes[i])
                reset_nested_counters(scopes[i], nested_scopes, scope_counters)


def get_names(x, ls):
    if isinstance(x, DataFlowGraph):
        ls.append(x.id)
        get_names(x.output[0], ls)
    elif isinstance(x, OpType):
        ls.append(x.id)
        for o in x.operands:
            get_names(o, ls)
    elif isinstance(x, TensorType):
        ls.append(x.id)


def get_correct_hierarchy_map(names):
    max_value = 0
    for name in names:
        s = re.findall(r"\{(\d*)\}", name)
        if s:
            for n in s:
                max_value = max(max_value, int(n))
    name_map = {}
    values = list(range(max_value + 1))
    values.reverse()

    for name in names:
        name_map[name] = name.format(*values)
    return name_map


def fix_hierarchy(dfg, name_map):
    if isinstance(dfg, DataFlowGraph):
        if dfg.id in name_map:
            dfg.id = name_map[dfg.id]
        fix_hierarchy(dfg.output[0], name_map)
    elif isinstance(dfg, OpType):
        if dfg.id in name_map:
            dfg.id = name_map[dfg.id]
        for o in dfg.operands:
            fix_hierarchy(o, name_map)
    elif isinstance(dfg, TensorType):
        if dfg.id in name_map:
            dfg.id = name_map[dfg.id]


def get_hierarchical_dict(x, hierarchy_dict, current_scope, inputs, scopes, input_names, nested_scopes, scope_counters, tensors):
    if isinstance(x, DataFlowGraph):
        if x in inputs:
            for i in inputs[x]:
                current_scope.remove(scopes[i])

        scopes[x] = x.id
        inp = x.inputs[0]

        if inp in inputs:
            inputs[inp].append(x)
        else:
            inputs[inp] = [x]

        current_scope += [scopes[x]]
        cur_scope_str = current_scope[-1]
        if cur_scope_str not in input_names:
            current_max = max(list(input_names.values()) + [0])
            input_names[cur_scope_str] = current_max

        current_dict_level = hierarchy_dict
        for scope in current_scope:
            if scope in current_dict_level:
                current_dict_level = current_dict_level[scope]
            else:
                current_dict_level[scope] = {}

        get_hierarchical_dict(x.output[0], hierarchy_dict, current_scope, inputs, scopes, input_names, nested_scopes, scope_counters, tensors)

    elif isinstance(x, OpType):
        current_dict_level = hierarchy_dict
        for scope in current_scope:
            if scope in current_dict_level:
                current_dict_level = current_dict_level[scope]

        current_dict_level[x.id] = []

        if x.id not in input_names:
            current_max = max(list(input_names.values()) + [-1])
            input_names[x.id] = current_max

        for o in x.operands:
            current_dict_level[x.id].append(o.id)
            if o.id not in input_names:
                current_max = max(list(input_names.values()) + [-1])
                input_names[o.id] = current_max + 1
                if isinstance(o, TensorType):
                    tensors[o.id] = current_max + 1

            cs = copy(current_scope)
            get_hierarchical_dict(o, hierarchy_dict, cs, inputs, scopes, input_names, nested_scopes, scope_counters, tensors)

        if x in inputs:
            for i in inputs[x]:
                current_scope.remove(scopes[i])

    elif isinstance(x, TensorType):
        if x in inputs:
            for i in inputs[x]:
                current_scope.remove(scopes[i])
