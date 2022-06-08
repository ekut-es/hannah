from hannah.nas.backend import TorchBackend
from hannah.nas.dataflow.dataflow_graph import dataflow
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.ops import axis, batched_image_tensor, float_t, tensor
from hannah.nas.parameters.parameters import (
    CategoricalParameter,
    FloatScalarParameter,
    IntScalarParameter,
)


@dataflow
def conv_relu(
    input: TensorType,
    output_channel=IntScalarParameter(4, 64),
    kernel_size=CategoricalParameter([1, 3, 5]),
    stride=CategoricalParameter([1, 2]),
):

    weight = tensor(
        (
            axis("o", size=output_channel),
            axis("i", size=input.output_tensor().axis["c"].size),
            axis("kh", size=kernel_size),
            axis("kw", size=kernel_size),
        ),
        dtype=float_t(),
        name="weight",
    )

    op = OpType("conv2d", input, weight, stride=stride)
    relu = OpType("relu", op)
    return relu


@dataflow
def block(
    input: TensorType,
    expansion=FloatScalarParameter(1, 6),
    output_channel=IntScalarParameter(4, 64),
    kernel_size=CategoricalParameter([1, 3, 5]),
    stride=CategoricalParameter([1, 2]),
):

    out = conv_relu(input, output_channel * expansion, kernel_size, stride)
    out = conv_relu(out, output_channel, 1, 1)
    return out


def test_convert_block():
    input = batched_image_tensor(name="input")
    out = block(input)
    network = block(out)

    backend = TorchBackend

    module = TorchBackend.build(network)

    # print(module)
    # print(module.code)
    breakpoint()


if __name__ == "__main__":
    test_convert_block()
