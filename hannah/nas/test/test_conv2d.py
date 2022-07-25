from hannah.nas.dataflow.dataflow_graph import dataflow
from hannah.nas.dataflow.ops import conv2d  # Import to load in registry
from hannah.nas.dataflow.registry import op
from hannah.nas.expressions.placeholder import DefaultInt
from hannah.nas.ops import batched_image_tensor, weight_tensor
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter


@dataflow
def conv2d(input, channel, kernel_size=DefaultInt(1), stride=DefaultInt(1), dilation=DefaultInt(1)):
    weight = weight_tensor(shape=(channel, input['c'], kernel_size, kernel_size), name='weight')
    padding = kernel_size // 2
    return op("Conv2d", input, weight, dilation=dilation, stride=stride, padding=padding)


@dataflow
def chained_convs(input, channel, kernel_size=DefaultInt(1), stride=DefaultInt(1), dilation=DefaultInt(1)):
    padding = kernel_size // 2
    weight1 = weight_tensor(shape=(channel, input['c'], kernel_size, kernel_size), name='weight')
    conv1 = op("Conv2d", input, weight1, dilation=dilation, stride=stride, padding=padding)

    weight2 = weight_tensor(shape=(channel, input['c'], kernel_size, kernel_size), name='weight')
    conv2 = op("Conv2d", conv1, weight2, dilation=dilation, stride=stride, padding=padding)

    return conv2


def test_conv2d():
    inp = batched_image_tensor(name="input")

    kernel_size = CategoricalParameter([1, 3, 5])
    stride = CategoricalParameter([1, 2])

    conv = conv2d(inp, channel=IntScalarParameter(4, 64), kernel_size=kernel_size, stride=stride)

    conv['conv2d.0.Conv2d.0'].kernel_size.set_current(3)
    conv['conv2d.0.Conv2d.0'].stride.set_current(2)

    returned_tensor = conv.output.output_tensor()
    for name, ax in returned_tensor.tensor_type.axis.items():
        print("{}: {}".format(name, ax.size.evaluate()))
    print()


def test_chained_conv2d():
    inp = batched_image_tensor(name="input")

    ks = CategoricalParameter([1, 3, 5])
    ks.set_current(3)
    convs = chained_convs(inp, channel=IntScalarParameter(4, 64), kernel_size=ks)
    returned_tensor = convs.output.output_tensor()

    for name, ax in returned_tensor.tensor_type.axis.items():
        print("{}: {}".format(name, ax.size.evaluate()))
    print()


if __name__ == '__main__':
    # test_conv2d()
    test_chained_conv2d()
    print()
