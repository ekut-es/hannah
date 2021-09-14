import torch.nn as nn


# base construct of a residual block
class ResBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, act_after_res=False, norm_after_res=False, norm_order=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.do_act = act_after_res
        self.do_norm = norm_after_res
        self.norm_order = norm_order
        # if the input channel count does not match the output channel count, apply skip to residual values
        self.apply_skip = self.in_channels != self.out_channels
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)
        # placeholders:
        self.blocks = nn.Identity()
        self.skip = nn.Identity()

    def forward(self, x):
        residual = x
        # do not use self.apply_skip for this: a skip connection may still be added to support elastic width
        # by default, skip is an Identity. It may be needlessly applied if the residual block implementation does not replace it with a skip or None
        if self.skip is not None:
            residual = self.skip(residual)
        x = self.blocks(x)
        x += residual
        # do activation and norm after applying residual (if enabled)
        if self.do_norm and self.norm_order.norm_before_act:
            x = self.norm(x)
        if self.do_act:
            x = self.act(x)
        if self.do_norm and self.norm_order.norm_after_act:
            x = self.norm(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def get_nested_modules(self):
        return nn.ModuleList([
            self.blocks,
            self.skip,
            self.norm,
            self.act
        ])


# residual block with a 1d skip connection
class ResBlock1d(ResBlockBase):
    def __init__(self, in_channels, out_channels, minor_blocks, act_after_res=False, norm_after_res=False, stride=1, norm_order=None):
        super().__init__(in_channels=in_channels, out_channels=out_channels, act_after_res=act_after_res, norm_after_res=norm_after_res, norm_order=norm_order)
        # set the minor block sequence if specified in construction
        # if minor_blocks is not None:
        self.blocks = minor_blocks
        # if applying skip to the residual values is required, create skip as a minimal conv1d
        # stride is also applied to the skip layer (if specified, default is 1)
        self.skip = nn.Sequential(
            nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(self.out_channels)
        )  # if self.apply_skip else None
        # as this does not know if an elastic width section may follow, the skip connection is required! it will be needed if the width is modified later
