from typing import List
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor import TensorTuple
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.dataflow.register_ops import add_op, add_shape_func


@add_op
class Sum:
    inputs: List[TensorExpression]


@add_shape_func("Sum")
def add_shape(op: OpType):
    tensors = []
    for operand in op.operands:
        tensors.append(operand.tensor_type())

    for tensor in tensors:
        assert tensors[0].dim() == tensor.dim()
    ax = []
    # constraints = []
    for tensor in tensors:
        for ax1, ax2 in zip(tensors[0].axis.values(), tensor.axis.values()):
            con = ax1.size == ax2.size
            # constraints.append(con)
            op.cond(con)
            # assert con.evaluate(), """Tensor axis sizes do not match: Axis {} with dimension
            #                        {} and axis {} with dimension {}""".format(ax1,
            #                                                                   input.tensor_type.axis[ax1].size,
            #                                                                   ax2,
            #                                                                   other.tensor_type.axis[ax2].size)
            ax.append(ax1.new())

    ax = tuple(ax)
    return TensorType(ax, dtype=tensors[0].dtype)
