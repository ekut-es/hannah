from hannah.nas.dataflow.data_type import IntType
from hannah.nas.dataflow.dataflow_graph import dataflow
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.expressions.placeholder import DefaultInt, UndefinedInt
from hannah.nas.ops import axis, batched_image_tensor, tensor
from hannah.nas.dataflow.registry import add_op, add_shape_func, add_conversion, op


@add_op
class Conv2d:
    input: TensorExpression
    weight: TensorExpression
    size: int
    dilation: int = 1
    stride : int = 1


# @add_shape_func("Conv2D")
# def conv2d_shape(op: OpType):
#     pass


# @add_conversion("Conv2D", target="torch")
# def conv2d_torch(op: OpType):
#     pass


@dataflow
def conv2d(input):
    kernel_size = UndefinedInt()
    stride = DefaultInt(1)
    dilation = DefaultInt(1)
    weight = tensor(
        (
            axis("o", UndefinedInt()),
            axis("i", UndefinedInt()),
            axis("kh", kernel_size),
            axis("kw", kernel_size),
        ),
        dtype=IntType(),
    )
    return op("Conv2D", input, weight, size=kernel_size, dilation=dilation, stride=stride)


if __name__ == '__main__':
    t1 = batched_image_tensor(name="input")
    t2 = batched_image_tensor(name="weight")
    # t3 = batched_image_tensor(name="additional_operand")

    conv = op("Conv2d", t1, t2, size=3, dilation=1)
    print()
