from hannah.nas.dataflow.dataflow_utils import find_first_op_in_dfg
from hannah.nas.dataflow.tensor_expression import TensorExpression


class OpType(TensorExpression):
    def __init__(self, *operands, tensor_type=None, name="", **attributes):
        super().__init__(*operands, tensor_type=tensor_type, name=name)
        self.attributes = attributes
        self.link_users()

    def link_users(self):
        for operand in self.operands:
            last_output = find_first_op_in_dfg(operand)
            last_output.users.append(self)
            # operand.users.append(self)

    def __repr__(self) -> str:
        # ret = ""
        # ret += "{}(".format(self.name) + \
        #        "".join(["\t{}, \n".format(o) for o in self.operands]) + \
        #        "".join(["\t{}={}".format(key, str(attr)) for key, attr in self.attributes.items()]) + \
        #        ")"
        # return ret
        return "Op({})".format(self.name)


# def register_scope(scope, scope_counters):
#     scope_counters[scope] = 1
#     return scope + ".{{{}}}".format(scope_counters[scope])
