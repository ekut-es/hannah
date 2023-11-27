from hannah.models.embedded_vision_net.expressions import expr_product
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter
from hannah.models.resnet.operators import dynamic_depth
from hannah.models.resnet.blocks import block, conv_relu_bn, classifier_head


def search_space(name, input, num_classes=10):
    out_channels = IntScalarParameter(16, 64, step_size=4, name='out_channels')
    kernel_size = CategoricalParameter([3, 5, 7, 9], name='kernel_size')
    stride = CategoricalParameter([1, 2], name='stride')

    depth = IntScalarParameter(0, 2, name='depth')
    num_blocks = IntScalarParameter(0, 6, name='num_blocks')

    stem_kernel_size = CategoricalParameter([3, 5], name="kernel_size")
    stem_channels = IntScalarParameter(min=16, max=32, step_size=4, name="out_channels")
    out = conv_relu_bn(input, stem_channels, stem_kernel_size, stride.new())

    exits = []
    for i in range(num_blocks.max+1):
        out = block(out,
                    depth=depth.new(),
                    out_channels=out_channels.new(),
                    kernel_size=kernel_size.new(),
                    stride=stride.new())
        exits.append(out)

    out = dynamic_depth(*exits, switch=num_blocks)
    out = classifier_head(out, num_classes=num_classes)

    strides = [v for k, v in out.parametrization(flatten=True).items() if k.split('.')[-1] == 'stride']
    total_stride = expr_product(strides)
    out.cond(input.shape()[2] / total_stride > 1)

    return out
