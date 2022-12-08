from hannah.nas.core.parametrized import is_parametrized
from hannah.nas.dataflow.dataflow_utils import find_first_op_in_dfg, find_leaf_nodes
from hannah.nas.dataflow.scoping_utils import get_id_and_update_counters, update_scope
from hannah.nas.dataflow.tensor_expression import TensorExpression
import hannah.nas.dataflow.registry as reg
from hannah.nas.parameters.parametrize import parametrize


@parametrize
class OpType(TensorExpression):
    def __init__(self, *operands, tensor_type=None, name="", **attributes):
        super().__init__(*operands, tensor_type=tensor_type, name=name)

        self._PARAMETERS = {}
        # self._conditions = []

        for i, operand in enumerate(operands):
            if is_parametrized(operand):
                self._PARAMETERS[i] = operand

        for name, attribute in attributes.items():
            setattr(self, name, attribute)
            if is_parametrized(attribute):
                self._PARAMETERS[name] = attribute

        self.attributes = attributes
        self.link_users()

    def next_backwards(self):
        return list(self.operands)

    def link_users(self):
        for operand in self.operands:
            last_output = find_first_op_in_dfg(operand)
            if self not in last_output.users:
                last_output.users.append(self)

    def set_scope(self, current_scope, counters, visited):
        current_scope = update_scope(self, current_scope)
        scope_id = get_id_and_update_counters(current_scope, counters)
        self.id = scope_id
        self.set_param_scopes()
        visited.append(self)

        leafs = []
        find_leaf_nodes(self, leafs, visited)

        while leafs:
            leaf = leafs.pop(-1)
            leaf.set_scope(current_scope, counters, visited)
            visited.append(leaf)

    def tensor_type(self):
        tensortype = reg.shape(self.name)(self)

        return tensortype

    # def sample(self):
    #     for _key, param in self._PARAMETERS.items():
    #         param.sample()

    # def set_current(self, value):
    #     self.set_params(**value)
    #     self.check(None)  # argument "value" not needed currently

    # def check(self, value):
    #     for con in self._conditions:
    #         if not con.evaluate():
    #             raise Exception("Condition not satisfied: {}".format(con))

    # def instantiate(self):
    #     instance = deepcopy(self)
    #     instance._parametrized = False
    #     self.check(None)

    #     for key, param in instance._PARAMETERS.items():
    #         instantiated_value = param.instantiate()
    #         instance._PARAMETERS[key] = instantiated_value
    #         setattr(instance, key, instantiated_value)
    #     return instance

    # def parameters(self):
    #     return self._PARAMETERS

    def convert(self, target):
        return reg.convert(self.name, target)(self)

    def __repr__(self) -> str:
        return "Op({})".format(self.id)
