from hannah.nas.dataflow.dataflow_utils import find_first_op_in_dfg, find_leaf_nodes
from hannah.nas.dataflow.scoping_utils import get_id_and_update_counters, update_scope
from hannah.nas.dataflow.tensor import Tensor
from hannah.nas.dataflow.tensor_expression import TensorExpression
import hannah.nas.dataflow.registry as reg
from hannah.nas.parameters.parameters import Parameter


class OpType(TensorExpression):
    def __init__(self, *operands, tensor_type=None, name="", **attributes):
        super().__init__(*operands, tensor_type=tensor_type, name=name)

        self._parameters = {}

        for name, attribute in attributes.items():
            setattr(self, name, attribute)
            if isinstance(attribute, Parameter):
                self._parameters[name] = attribute

        self.attributes = attributes
        self.link_users()
        print()

    def link_users(self):
        for operand in self.operands:
            last_output = find_first_op_in_dfg(operand)
            last_output.users.append(self)

    def set_scope(self, current_scope, counters, visited):
        current_scope = update_scope(self, current_scope)
        scope_id = get_id_and_update_counters(current_scope, counters)
        self.id = scope_id
        visited.append(self)

        leafs = []
        find_leaf_nodes(self, leafs, visited)

        while leafs:
            leaf = leafs.pop(-1)
            leaf.set_scope(current_scope, counters, visited)
            visited.append(leaf)

    def output_tensor(self):
        tensortype = reg.shape(self.name)(self)
        return Tensor(tensor_type=tensortype, name=self.id + '.output')

    def convert(self, target):
        return reg.convert(self.name, target)(self)

    def __repr__(self) -> str:
        return "Op({})".format(self.id)
