
from hannah.nas.dataflow.dataflow_graph import dataflow
from hannah.nas.dataflow.dataflow_utils import process_int
from hannah.nas.expressions.placeholder import DefaultInt
from hannah.nas.ops import tensor_by_tuples, weight_tensor
from hannah.nas.dataflow.ops import add  # import to register
from hannah.nas.dataflow.registry import op
from hannah.nas.parameters.parameters import CategoricalParameter


def test_add():
    tensor_a = tensor_by_tuples((1, 4, 16, 16), ('a', 'b', 'c', 'd'))
    tensor_b = tensor_by_tuples((1, 4, 16, 16), ('a', 'b', 'c', 'd'))

    add_op = op('Add', tensor_a, tensor_b)
    returned_tensor = add_op.output_tensor()
    for name, ax in returned_tensor.tensor_type.axis.items():
        print("{}: {}".format(name, ax.size.evaluate()))
    print()


@dataflow
def parallel_convs(input):
    channel = process_int(32)
    kernel_size = CategoricalParameter([1, 3, 5])
    stride = CategoricalParameter([1, 2])
    dilation = DefaultInt(1)

    input_tensor = input.output_tensor()

    weight1 = weight_tensor(shape=(channel, input_tensor['c'], kernel_size, kernel_size), name='weight')
    conv1 = op("Conv2d", input, weight1, dilation=dilation, stride=stride)

    weight2 = weight_tensor(shape=(channel, input_tensor['c'], kernel_size, kernel_size), name='weight')
    conv2 = op("Conv2d", input, weight2, dilation=dilation, stride=stride)

    add_op = op('Add', conv1, conv2)

    return add_op


def test_parallel_convs():
    input_tensor = tensor_by_tuples((1, 3, 16, 16), ('n', 'c', 'h', 'w'), name='input')
    convs = parallel_convs(input_tensor)

    # convs['parallel_convs.0.Conv2d.0'].stride.set_current(2)

    returned_tensor = convs.output.output_tensor()

    for name, ax in returned_tensor.tensor_type.axis.items():
        print("{}: {}".format(name, ax.size.evaluate()))
    print()


if __name__ == '__main__':
    test_add()
    test_parallel_convs()
    print()
