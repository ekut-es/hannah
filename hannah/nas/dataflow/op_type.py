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

    def get_hierarchical_dict(self, hierarchy_dict, current_scope, inputs, scopes, input_names, nested_scopes, scope_counters, tensors):
        current_dict_level = hierarchy_dict
        for scope in current_scope:
            if scope in current_dict_level:
                current_dict_level = current_dict_level[scope]

        current_dict_level[self.id] = []

        if self.id not in input_names:
            current_max = max(list(input_names.values()) + [-1])
            input_names[self.id] = current_max

        for o in self.operands:
            current_dict_level[self.id].append(o.id)
            if o.id not in input_names:
                current_max = max(list(input_names.values()) + [-1])
                input_names[o.id] = current_max + 1
                if isinstance(o, TensorType):
                    tensors[o.id] = current_max + 1

            cs = copy(current_scope)
            o.get_hierarchical_dict(hierarchy_dict, cs, inputs, scopes, input_names, nested_scopes, scope_counters, tensors)

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
