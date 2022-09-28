import logging

import torch.nn as nn

from ..utilities import flatten_module_list

# base construct of a residual block
from .elasticBase import ElasticBase1d
from .elasticBatchnorm import ElasticWidthBatchnorm1d
from .elasticchannelhelper import ElasticChannelHelper
from .elastickernelconv import ElasticConvBnReLu1d
from .elasticquantkernelconv import ElasticQuantConvBnReLu1d


counter_res = 0
time_to_break = 1260
time_to_break = 1299

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
        self.blocks = nn.Identity()
        self.skip = nn.Identity()

    def forward(self, x):
        global counter_res
        global time_to_break

        ElasticBase1d.res_break = counter_res == time_to_break
        residual = x
        # do not use self.apply_skip for this: a skip connection may still be
        # added to support elastic width
        # by default, skip is an Identity. It may be needlessly applied if the
        # residual block implementation does not replace it with a skip or None
        if self.skip is not None:
            residual = self.skip(residual)
        try:
            x = self.blocks(x)
        except RuntimeError as r:
            logging.warn(r)
            for _, actualModel in self.blocks._modules.items():
                logging.info(f"XKA Module List: {actualModel}")
                logging.info(
                    f"XKA Settings: oc={actualModel.out_channels}, ic={actualModel.in_channels}, weights={actualModel.weight.shape}, k={actualModel.kernel_size}, s={actualModel.stride}, g={actualModel.groups}"
                )

        # logging.debug(f"Shape input: {x.shape} , Shape residual: {residual.shape}")
        x += residual
        if self.do_act:
            x = self.act(x)
        counter_res += 1
        return x

    def get_nested_modules(self):
        return nn.ModuleList([self.blocks, self.skip, self.act])


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

        # MR 20220622
        # TODO vereinheitlichen - still necessary ?
        for _, block in minor_blocks._modules.items():
            for _, actualModel in block._modules.items():
                logging.info(f"XKA Module List: {actualModel}")
                if isinstance(actualModel, ElasticBase1d):
                    logging.info(
                        f"XKA Settings: oc={actualModel.out_channels}, ic={actualModel.in_channels}, weights={actualModel.weight.shape}, k={actualModel.kernel_size}, s={actualModel.stride}, g={actualModel.groups}"
                    )
        self.norm = ElasticWidthBatchnorm1d(out_channels)
        self.act = nn.ReLU()
        self.qconfig = qconfig
        self.quant_skip = quant_skip
        # if applying skip to the residual values is required, create skip as a minimal conv1d
        # stride is also applied to the skip layer (if specified, default is 1)
        if not quant_skip:
            self.skip = nn.Sequential(
                ElasticConvBnReLu1d(
                    self.in_channels,
                    out_channels,
                    kernel_sizes=[1],
                    dilation_sizes=[1],
                    groups=[1],
                    dscs=[False],
                    stride=stride,
                    bias=False,
                    out_channel_sizes=flatten_module_list(self.blocks)[
                        -1
                    ].out_channel_sizes,
                    # TODO to delete after ana
                    from_skipping=True
                ),
            )
        else:
            self.skip = nn.Sequential(
                ElasticQuantConvBnReLu1d(
                    self.in_channels,
                    out_channels,
                    kernel_sizes=[1],
                    dilation_sizes=[1],
                    stride=stride,
                    groups=[1],
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
