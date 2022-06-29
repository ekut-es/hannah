from hannah.nas.dataflow.dataflow_graph import dataflow
from hannah.nas.dataflow.ops import conv2d  # Import to load in registry
from hannah.nas.dataflow.registry import op
from hannah.nas.expressions.placeholder import DefaultInt
from hannah.nas.ops import batched_image_tensor, weight_tensor
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter


@dataflow
def conv(input, channel, kernel_size=DefaultInt(1), stride=DefaultInt(1), dilation=DefaultInt(1)):
    weight = weight_tensor(shape=(channel, input['c'], kernel_size, kernel_size), name='weight')
    padding = kernel_size // 2
    return op("Conv2d", input, weight, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding)


def test_conv2d():
    inp = batched_image_tensor(name="input")

    kernel_size = CategoricalParameter([1, 3, 5])
    stride = CategoricalParameter([1, 2])
    channel = IntScalarParameter(4, 64)
    channel.set_current(52)
    conv_dataflow = conv(inp, channel=channel, kernel_size=kernel_size, stride=stride)

    torch_op = conv_dataflow['conv.0.Conv2d.0'].convert(target='torch')
    print()


if __name__ == '__main__':
    test_conv2d()
    print()