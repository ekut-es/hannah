from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter
from hannah.models.conv_vit.operators import dynamic_depth
from hannah.models.conv_vit.blocks import stem, block, classifier_head
from hannah.nas.functional_operators.op import search_space


@search_space
def conv_vit(name, input, num_classes=10, use_lin_attn=False):
    # Stem parameters
    stem_kernel_size = CategoricalParameter([3, 5, 7, 9], name="kernel_size")
    stem_stride = CategoricalParameter([1, 2], name='stride')
    stem_channels = IntScalarParameter(min=16, max=32, step_size=4, name="out_channels")
    stem_channel_ratio = IntScalarParameter(1, 4, name='expand_ratio')
    stem_pool_size = CategoricalParameter([3, 5, 7, 9], name='kernel_size')

    # Block parameters
    out_channels = IntScalarParameter(16, 64, step_size=4, name='out_channels')
    kernel_size = CategoricalParameter([3, 5, 7, 9], name='kernel_size')
    stride = CategoricalParameter([1, 2], name='stride')
    channel_ratio = IntScalarParameter(1, 2, name='expand_ratio')

    num_heads = IntScalarParameter(2, 8, step_size=2, name='num_heads')
    d_model = IntScalarParameter(16, 64, step_size=16, name='d_model')

    # Depth & Number of blocks
    depth = IntScalarParameter(0, 2, name='depth')
    num_blocks = IntScalarParameter(0, 4, name='num_blocks')

    # Stem
    out = stem(
        input,
        out_channels=stem_channels,
        kernel_size=stem_kernel_size,
        stride=stem_stride,
        channel_ratio=stem_channel_ratio,
        pool_size=stem_pool_size
    )

    # Blocks
    exits = []
    for _ in range(num_blocks.max+1):
        out = block(
            out,
            depth=depth.new(),
            out_channels=out_channels.new(),
            kernel_size=kernel_size.new(),
            stride=stride.new(),
            channel_ratio=channel_ratio.new(),
            num_heads=num_heads.new(),
            d_model=d_model.new(),
            use_lin_attn=use_lin_attn,
        )
        exits.append(out)

    out = dynamic_depth(*exits, switch=num_blocks)
    output_fmap = out.shape()[2]
    out = classifier_head(out, num_classes=num_classes)

    stride_params = [v for k, v in out.parametrization(flatten=True).items() if k.split('.')[-1] == 'stride']
    out.cond(output_fmap > 1, allowed_params=stride_params)

    return out
