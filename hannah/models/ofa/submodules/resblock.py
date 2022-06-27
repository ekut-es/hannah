import torch.nn as nn

# base construct of a residual block
from ..utilities import flatten_module_list
from .elasticBatchnorm import ElasticWidthBatchnorm1d
from .elasticchannelhelper import ElasticChannelHelper, SequenceDiscovery
from .elastickernelconv import ElasticConv1d, ElasticConvBn1d
from .elasticquantkernelconv import ElasticQuantConv1d, ElasticQuantConvBn1d


class ResBlockBase(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        act_after_res=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.do_act = act_after_res
        # if the input channel count does not match the output channel count,
        # apply skip to residual values
        self.apply_skip = self.in_channels != self.out_channels
        # placeholders:
        self.act = nn.Identity()
        self.norm = nn.Identity()
        self.blocks = nn.Identity()
        self.skip = nn.Identity()

    def forward(self, x):
        residual = x
        # do not use self.apply_skip for this: a skip connection may still be
        # added to support elastic width
        # by default, skip is an Identity. It may be needlessly applied if the
        # residual block implementation does not replace it with a skip or None
        if self.skip is not None:
            residual = self.skip(residual)
        x = self.blocks(x)
        x += residual
        if self.do_act:
            x = self.act(x)
        return x

    def get_nested_modules(self):
        return nn.ModuleList([self.blocks, self.skip, self.norm, self.act])


# residual block with a 1d skip connection
class ResBlock1d(ResBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        minor_blocks,
        act_after_res=True,
        stride=1,
        norm_before_act=True,
        quant_skip=False,
        qconfig=None,
        out_quant=True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            act_after_res=act_after_res,
        )
        # set the minor block sequence if specified in construction
        # if minor_blocks is not None:
        self.blocks = minor_blocks
        self.norm = ElasticWidthBatchnorm1d(out_channels)
        self.act = nn.ReLU()
        self.qconfig = qconfig
        self.quant_skip = quant_skip
        # if applying skip to the residual values is required, create skip as a minimal conv1d
        # stride is also applied to the skip layer (if specified, default is 1)
        if not quant_skip:
            self.skip = nn.Sequential(
                ElasticConvBn1d(
                    self.in_channels,
                    out_channels,
                    kernel_sizes=[1],
                    dilation_sizes=[1],
                    stride=stride,
                    bias=False,
                    out_channel_sizes=flatten_module_list(self.blocks)[
                        -1
                    ].out_channel_sizes,
                ),
            )
        else:
            self.skip = nn.Sequential(
                ElasticQuantConvBn1d(
                    self.in_channels,
                    out_channels,
                    kernel_sizes=[1],
                    dilation_sizes=[1],
                    stride=stride,
                    bias=False,
                    qconfig=qconfig,
                    out_channel_sizes=flatten_module_list(self.blocks)[
                        -1
                    ].out_channel_sizes,
                ),
            )  # if self.apply_skip else None
        if self.qconfig is not None:
            self.activation_post_process = (
                self.qconfig.activation() if out_quant else nn.Identity()
            )
        # as this does not know if an elastic width section may follow,
        # the skip connection is required! it will be needed if the width is modified later

    def forward(self, x):
        if isinstance(x, SequenceDiscovery):
            # DISCOVER BLOCKS FIRST, THEN SKIP.
            # this will set primary targets correctly, without needing to specify
            # the skip as a secondary target in discover explicitly.
            second_discovery = x.split()
            blocks_resulting_discovery = self.blocks.forward(x)
            skip_resulting_discovery = self.skip.forward(second_discovery)
            # merge the two discoveries together where the module outputs would normally be added.
            new_discovery = blocks_resulting_discovery.merge_sequence_discovery(
                skip_resulting_discovery
            )
            # pass it through the resblock internal norm - norm/act sequence is irrelevant
            # as the activation does not affect discovery whatsoever.
            return self.norm(new_discovery)
        else:
            output = super().forward(x)
            if self.qconfig is not None:
                return self.activation_post_process(output)
            return output

    def get_input_layer(self):
        input = nn.ModuleList()
        input.append(flatten_module_list(self.skip)[0])
        input.append(flatten_module_list(self.blocks)[0])
        return input

    def get_output_layer(self):
        output = nn.ModuleList()
        output.append(flatten_module_list(self.skip)[-1])
        output.append(flatten_module_list(self.blocks)[-1])
        return output

    def create_internal_channelhelper(self):
        output = nn.ModuleList()

        for idx in range(len(self.blocks) - 1):
            if len(self.blocks[idx].out_channel_sizes) > 1:
                ech = ElasticChannelHelper(self.blocks[idx].out_channel_sizes)
                ech.add_source_item(self.blocks[idx])
                ech.add_targets(self.blocks[idx + 1])
                output.append(ech)

        return output
