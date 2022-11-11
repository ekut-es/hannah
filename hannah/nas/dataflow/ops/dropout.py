from copy import deepcopy
from typing import Union
from hannah.nas.core.expression import Expression
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.dataflow.register_ops import add_op, add_shape_func
from hannah.nas.expressions.placeholder import FloatRange
from hannah.nas.parameters.parameters import FloatScalarParameter


@add_op
class Dropout2d:
    input: TensorExpression
    p: Expression


@add_shape_func("Dropout2d")
def add_shape(op: OpType):
    return deepcopy(op.operands[0].tensor_type())
