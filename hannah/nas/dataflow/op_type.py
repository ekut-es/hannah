from copy import copy
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.dataflow.dataflow_utils import reset_nested_counters


class OpType:
    def __init__(self, name, *operands, **attributes):
        self.name = name
        self.id = name
        self.operands = list(operands)
        self.attributes = attributes

    def output_tensor(self):
        # TODO: what happens with multiple operands?
        # => here one has to define how the op
        # changes the tensor
        return self.operands[0].output_tensor()

    def dfg_line_representation(self, indent, input_names):
        result_and_name = '\t'*indent + '%{{{}}} = {}('.format(input_names[self], self.id)
        operands = ', '.join(['%{{{}}}' for _ in range(len(self.operands))]).format(*[input_names[x] for x in self.operands]) + ' '
        attributes = ', '.join(['{}={}'.format(k, v) for k, v in self.attributes.items()])
        suffix = ')'
        return result_and_name + operands + attributes + suffix

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
            e.g. [block.0, block.0.conv.0, ... ]
        inputs : dict
            mapping of node -> list of nodes that this node is an input of
            e.g. if out = block1(block0) => {DataFlowGraph(name=block0): [DataFlowGraph(name=block1)]}
        scopes : dict
            mapping of node -> scope name/id
        input_names : dict
            mapping of node -> int counter for input
            different scopes can have the same int-input representation because
            the inputs "trickle down": e.g.
            {block.1: 0, block.1.conv_relu.1: 0, block.1.conv_relu.1.relu: 0}
            -> 'block.1' encapsulates 'block.1.conv_relu.1' and that encapsulates
               'block.1.conv_relu.1.relu', and so the input tensor is passed down the scope
        tensors : list
            list of input tensors collected during traversal for more convenient
            listing later on
        """

        # expand the hierarchy dict and navigate to current level
        current_dict_level = hierarchy_dict
        for scope in current_scope:
            if scope in current_dict_level:
                current_dict_level = current_dict_level[scope]
        current_dict_level[self] = []

        # extract input/output int representation for dataflow printing
        if self not in input_names:
            current_max = max(list(input_names.values()) + [-1])
            input_names[self] = current_max

        for o in self.operands:
            current_dict_level[self].append(o)
            if o not in input_names:
                current_max = max(list(input_names.values()) + [-1])
                input_names[o] = current_max + 1
                if isinstance(o, TensorType):
                    tensors[o] = current_max + 1

            cs = copy(current_scope)
            o.get_hierarchical_dict(hierarchy_dict, cs, inputs, scopes, input_names, tensors)

    def insert_scope_to_id(self, inputs, scopes, current_scope, scope_counters, nested_scopes):
        self.id = ".".join(current_scope) + ".{}".format(self.name)

        for o in self.operands:
            cs = copy(current_scope)
            o.insert_scope_to_id(inputs, scopes, cs, scope_counters, nested_scopes)

        if self in inputs:
            for i in inputs[self]:
                current_scope.remove(scopes[i])
                reset_nested_counters(scopes[i], nested_scopes, scope_counters)

    def __repr__(self) -> str:
        ret = ""
        ret += "{}(".format(self.name) + \
               "".join(["\t{}, \n".format(o) for o in self.operands]) + \
               "".join(["\t{}={}".format(key, str(attr)) for key, attr in self.attributes.items()]) + \
               ")"
        return ret
