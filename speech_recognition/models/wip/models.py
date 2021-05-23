import torch.nn as nn
import numpy as np
# import torch

# from ..utils import ConfigType, SerializableModule


# Conv1d with automatic padding for the set kernel size
def conv1d_auto_padding(conv1d: nn.Conv1d):
    conv1d.padding = conv1d.kernel_size[0] // 2
    return conv1d


# base construct of a residual block
class ResBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, act_after_res=True, norm_after_res=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.do_act = act_after_res
        self.do_norm = norm_after_res
        # if the input channel count does not match the output channel count, apply skip to residual values
        self.apply_skip = self.in_channels != self.out_channels
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)
        # placeholders:
        self.blocks = nn.Identity()
        self.skip = nn.Identity()

    def forward(self, x):
        residual = x
        if self.apply_skip:
            residual = self.skip(residual)
        x = self.blocks(x)
        x += residual
        # do activation and norm after applying residual (if enabled)
        if self.do_act:
            x = self.act(x)
        if self.do_norm:
            x = self.norm(x)

    def __call__(self, x):
        return self.forward(x)


# residual block with a 1d skip connection
class ResBlock1d(ResBlockBase):
    def __init__(self, in_channels, out_channels, minor_blocks=None, act_after_res=True, norm_after_res=True):
        super().__init__(in_channels=in_channels, out_channels=out_channels, act_after_res=act_after_res, norm_after_res=norm_after_res)
        # set the minor block sequence if specified in construction
        if minor_blocks is not None:
            self.blocks = minor_blocks
        # if applying skip to the residual values is required, create skip as a minimal conv1d
        self.skip = nn.Sequential(
            nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(self.out_channels)
        ) if self.apply_skip else None


def create(name: str, labels: int, input_shape, conv=[]):
    # print("################ SHAPE ################")
    # print(np.shape(input_shape))
    # print(input_shape)
    # print("################ SHAPE ################")
    flatten_n = input_shape[0]
    in_channels = input_shape[1]
    pool_n = input_shape[2]
    # the final output channel count is given by the last minor block of the last major block
    final_out_channels = conv[-1].blocks[-1].out_channels
    # get the max depth from the count of major blocks
    model = WIPModel(max_depth=len(conv), labels=labels, pool_kernel=pool_n, flatten_dims=flatten_n, out_channels=final_out_channels)
    model.block_config = conv
    next_in_channels = in_channels
    # breakpoint()

    for block_config in conv:
        if block_config.target == "forward":
            major_block = create_forward_block(blocks=block_config.blocks, in_channels=next_in_channels)
        elif block_config.target == "residual1d":
            major_block = create_residual_block_1d(blocks=block_config.blocks, in_channels=next_in_channels)
        else:
            raise Exception(f"Undefined target selected for major block: {block_config.target}")
        # output channel count of the last minor block will be the input channel count of the next major block
        next_in_channels = block_config.blocks[-1].out_channels
        model.conv_layers.append(major_block)

    return model


# build a sequence from a list of minor block configurations
def create_minor_block_sequence(blocks, in_channels):
    next_in_channels = in_channels
    minor_block_sequence = nn.ModuleList([])
    for block_config in blocks:
        if block_config.target == "conv1d":
            out_channels = block_config.out_channels
            # create a minor block, potentially with activation and norm
            minor_block_internal_sequence = nn.ModuleList([])
            minor_block_internal_sequence.append(conv1d_auto_padding(nn.Conv1d(
                    kernel_size=block_config.kernel_size,
                    in_channels=next_in_channels,
                    out_channels=out_channels
            )))
            if block_config.act:
                # add relu activation if act is set
                minor_block_internal_sequence.append(nn.ReLU())
            if block_config.norm:
                # add a batch norm if norm is set
                minor_block_internal_sequence.append(nn.BatchNorm1d(out_channels))

            minor_block_sequence.append(nn.Sequential(*minor_block_internal_sequence))
            # the input channel count of the next minor block is the output channel count of the previous block
            next_in_channels = out_channels
        # if an unknown target is selected for a minor block, throw an exception.
        else:
            raise Exception(f"Undefined target selected in minor block sequence: {block_config.target}")

    return nn.Sequential(*minor_block_sequence)


# build a basic forward major block
def create_forward_block(blocks, in_channels):
    return create_minor_block_sequence(blocks, in_channels)


# build a residual major block
def create_residual_block_1d(blocks, in_channels):
    minor_blocks = create_minor_block_sequence(blocks, in_channels)
    # the output channel count of the residual major block is the output channel count of the last minor block
    out_channels = blocks[-1].out_channels
    residual_block = ResBlock1d(in_channels=in_channels, out_channels=out_channels, minor_blocks=minor_blocks)
    return residual_block


class WIPModel(nn.Module):
    def __init__(self, max_depth: int, labels: int, pool_kernel: int, flatten_dims: int, out_channels: int):
        super().__init__()
        self.max_depth = max_depth
        self.active_depth = self.max_depth
        self.labels = labels
        self.pool_kernel = pool_kernel
        self.flatten_dims = flatten_dims
        self.out_channels = out_channels
        self.block_config = []
        # from convolution result dims
        self.pool = nn.AvgPool1d(pool_kernel)
        self.flatten = nn.Flatten(flatten_dims)
        self.linear = nn.Linear(out_channels, labels)  # from channel count at output

        self.conv_layers = nn.ModuleList([])

    # filter from DynamicConv2d
    def get_active_filter(self, out_channel, in_channel):
        # out_channels, in_channels/groups, kernel_size[0], kernel_size[1]
        return self.conv.weight[:out_channel, :in_channel, :]

    # def forward(self, x, out_channel=None):
    def forward(self, x):
        print("forwarding")
        print(np.shape(x))
        for layer in self.conv_layers[:self.active_depth]:
            """ # REMOVE: use __call__ to run forward(x)
            # if the layer has a forward function, use it.
            forward_function = getattr(layer, "forward", None)
            if callable(forward_function):
                x = layer.forward(x)
            # otherwise, apply the layer directly
            else:
               x = layer(x)
            """
            print("################ LAYER ################")
            print(np.shape(layer))
            print("################ LAYER ################")
            breakpoint()
            x = layer(x)

        result = x
        print(np.shape(result))
        result = self.pool(result)
        result = self.flatten(result)
        result = self.linear(result)
        # print(np.shape(result))

        return result
        # return nn.functional.conv2d(x, self.conv.weight, None, self.conv.stride, padding, self.conv.dilation, 1)

    def sample_active_subnet(self):
        return  # disabled until forward is fixed. active_depth will be default max_depth
        self.active_depth = np.random.randint(1, self.max_depth)
        # the new out channel count is given by the last minor block of the last active major block
        self.out_channels = self.block_config[:self.active_depth][-1].blocks[:-1].out_channels
        self.linear = nn.Linear(self.out_channels, self.labels)
        # print("Picked active depth: ", self.active_depth)
