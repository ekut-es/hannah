from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter
from hannah.models.conv_vit.operators import dynamic_depth
from hannah.models.conv_vit.blocks import stem, block, classifier_head


def search_space(name, input, num_classes=10):
    # Stem parameters
    stem_kernel_size = CategoricalParameter([3, 5, 7, 9], name="kernel_size")
    stem_stride = CategoricalParameter([1, 2], name='stride')
    stem_channels = IntScalarParameter(min=16, max=32, step_size=4, name="out_channels")

    # Block parameters
    kernel_size = CategoricalParameter([3, 5, 7, 9], name='kernel_size')
    stride = CategoricalParameter([1, 2], name='stride')

    num_heads = IntScalarParameter(2, 8, step_size=2, name='num_heads')
    d_model = IntScalarParameter(16, 64, step_size=16, name='d_model')
    expand_ratio = IntScalarParameter(1, 2, name='expand_ratio')
    out_channels = IntScalarParameter(16, 64, step_size=4, name='out_channels')

    depth = IntScalarParameter(0, 2, name='depth')
    num_blocks = IntScalarParameter(0, 4, name='num_blocks')

    # Stem
    out = stem(input, stem_kernel_size, stem_stride, stem_channels)

    # Blocks
    exits = []
    for _ in range(num_blocks.max+1):
        out = block(
            out,
            depth=depth.new(),
            expand_ratio=expand_ratio.new(),
            kernel_size=kernel_size.new(),
            stride=stride.new(),
            num_heads=num_heads.new(),
            d_model=d_model.new(),
            out_channels=out_channels.new()
        )
        exits.append(out)

    out = dynamic_depth(*exits, switch=num_blocks)
    output_fmap = out.shape()[2]
    out = classifier_head(out, num_classes=num_classes)

    stride_params = [v for k, v in out.parametrization(flatten=True).items() if k.split('.')[-1] == 'stride']
    out.cond(output_fmap > 1, allowed_params=stride_params)

    return out
