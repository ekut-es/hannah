from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.dataflow.register_ops import add_op, add_shape_func

from hannah.nas.dataflow.tensor_expression import TensorExpression


@add_op
class Add:
    input: TensorExpression
    other: TensorExpression


@add_shape_func("Add")
def add_shape(op: OpType):
    input = op.operands[0].tensor_type()
    other = op.operands[1].tensor_type()

    assert input.dim() == other.dim()
    ax = []
    # constraints = []
    for ax1, ax2 in zip(input.axis.values(), other.axis.values()):
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
    return TensorType(ax, dtype=input.dtype)
