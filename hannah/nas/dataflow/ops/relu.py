from copy import deepcopy
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.dataflow.register_ops import add_op, add_shape_func


@add_op
class Relu:
    input: TensorExpression


@add_shape_func("Relu")
def add_shape(op: OpType):
    return deepcopy(op.operands[0].tensor_type())
