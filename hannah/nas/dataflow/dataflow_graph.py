from typing import Iterable
import re
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.optional_op import OptionalOp
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.dataflow.dataflow_utils import register_scope, reset_nested_counters, scope_nester


class DataFlowGraph:
    def __init__(self, inputs, output, name: str = "dataflow") -> None:
        self.inputs = inputs
        self.output = output
        self.name = name
        self.id = name
        self.insert_scope_to_id({}, {}, [], {}, {})
        names = []
        get_names(self, names)
        name_map = get_correct_hierarchy_map(names)
        fix_hierarchy(self, name_map)
        # self.leaf_nodes = collect_leaf_nodes(self.output)

    def __getitem__(self, key):
        return getattr(self, key)

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
        self.get_hierarchical_dict(hierarchy_dict=hierarchy_dict,
                                   current_scope=[],
                                   inputs={},
                                   scopes={},
                                   input_names=input_names,
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

    def insert_scope_to_id(self, inputs, scopes, current_scope, scope_counters, nested_scopes):
        if self in inputs:
            for i in inputs[self]:
                current_scope.remove(scopes[i])
                reset_nested_counters(scopes[i], nested_scopes, scope_counters)
        scopes[self] = register_scope(self.name, scope_counters)
        self.id = ".".join(current_scope) + ".{}".format(scopes[self]) if current_scope else scopes[self]
        inp = self.inputs[0]
        if inp in inputs:
            inputs[inp].append(self)
        else:
            inputs[inp] = [self]
        nested_scopes = scope_nester(scopes[self], current_scope, nested_scopes)
        current_scope += [scopes[self]]
        # TODO: Handle multiple outputs
        self.output[0].insert_scope_to_id(inputs, scopes, current_scope, scope_counters, nested_scopes)

    def get_hierarchical_dict(self, hierarchy_dict, current_scope, inputs, scopes, input_names, tensors):
        """Recursively extract a dict that describes the
        scope hierarchy

        Parameters
        ----------
        hierarchy_dict : dict
            describes the scope hierarchy
            e.g. {'block.0': {'block.0.conv.0': ..., 'block.0.conv.1': ...}, 'block.1': ...}
        current_scope : list
            list of current scopes in descending order
            e.g. ['block.0', 'block.0.conv.0', ... ]
        inputs : dict
            mapping of node -> list of nodes that this node is an input of
            e.g. if out = block1(block0) => {DataFlowGraph(name=block0): [DataFlowGraph(name=block1)]}
        scopes : dict
            mapping of node -> scope name/id
        input_names : dict
            mapping of node_id -> int counter for input
            different scopes can have the same int-input representation because
            the inputs "trickle down": e.g.
            {'block.1': 0, 'block.1.conv_relu.1': 0, 'block.1.conv_relu.1.relu': 0}
            -> 'block.1' encapsulates 'block.1.conv_relu.1' and that encapsulates
               'block.1.conv_relu.1.relu', and so the input tensor is passed down the scope
        tensors : list
            list of input tensors collected during traversal for more convenient
            listing later on
        """
        # e.g. if out = block2(block_1), we know that the scope of block2 must end when we reach block1
        if self in inputs:
            for i in inputs[self]:
                current_scope.remove(scopes[i])

        scopes[self] = self.id
        # TODO: Support multiple inputs
        inp = self.inputs[0]

        # TODO: move to extra class
        if inp in inputs:
            inputs[inp].append(self)
        else:
            inputs[inp] = [self]

        current_scope += [scopes[self]]
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

        self.output[0].get_hierarchical_dict(hierarchy_dict, current_scope, inputs, scopes, input_names, tensors)

    def __str__(self) -> str:
        return self.get_string()

    def __repr__(self) -> str:
        return "DataFlowGraph(id={})".format(self.id)


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
