import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t, _size_any_opt_t
from typing import Union
from torch import Tensor


class Add(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *seq):
        args = seq[0]
        if len(args) == 2:
            return torch.add(*args)
        elif len(args) > 2:
            return torch.sum(torch.stack(args), dim=0)
        else:
            return args


class Input(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if self.in_channels != self.out_channels and stride == 1:
            self.relu = nn.ReLU(inplace=False)
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, 1, stride=1, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(self.out_channels, affine=True)
        elif stride == 2:
            assert self.out_channels % 2 == 0
            self.relu = nn.ReLU(inplace=False)
            self.conv_1 = nn.Conv2d(
                self.in_channels, self.out_channels // 2, 1, stride=stride, padding=0, bias=False
            )
            self.conv_2 = nn.Conv2d(
                self.in_channels, self.out_channels // 2, 1, stride=stride, padding=0, bias=False
            )
            self.bn = nn.BatchNorm2d(self.out_channels, affine=True)

    def forward(self, *seq):
        if self.stride == 2:
            x = self.relu(seq[0])
            if x.shape[2] % 2 == 1 or x.shape[3] % 2 == 1:
                x = F.pad(x, (0, x.shape[3] % 2, 0, x.shape[2] % 2), 'constant', 0)
            out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
            # print('Outshape', out.shape)
            out = self.bn(out)
            return out
        if self.in_channels != self.out_channels:
            out = self.relu(seq[0])
            out = self.conv(out)
            out = self.bn(out)
        else:
            out = seq[0]
        return out


class Concat(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *seq):
        args = seq[0]
        if isinstance(args, torch.Tensor):
            return args
        return torch.cat(args, dim=1)


class Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)


class Conv2dAct(Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 act_func=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        if act_func:
            self.act_fun = act_func
        else:
            self.act_fun = nn.Identity()

    def forward(self, x):
        out = super().forward(x)
        return self.act_fun(out)


class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Linear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self,
                 num_features,
                 eps=0.00001,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 device=None,
                 dtype=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)


class ReLU6(nn.ReLU6):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)


class Identity(nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        device = x.get_device() if x.is_cuda else torch.device('cpu')
        # noinspection PyUnresolvedReferences
        padding = torch.zeros(n, c, h, w, device=device, requires_grad=False)
        return padding


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size: _size_any_opt_t) -> None:
        super().__init__(output_size)

    def forward(self, input: Tensor) -> Tensor:
        out = super().forward(input)
        return out


class MixedOp(nn.Module):
    def __init__(self, ops, choice, mask=None):
        super().__init__()
        self.op = ops[choice]

    def forward(self, x):
        return self.op(x)

    def make_differentiable(self):
        raise NotImplementedError

    def prepare_shared_weight(self):
        raise NotImplementedError


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True):
        super(FactorizedReduce, self).__init__()
        if stride == 2:
            self.is_identity = False
            assert C_out % 2 == 0
            self.relu = nn.ReLU(inplace=False)
            self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
            self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(C_out, affine=affine)
        elif C_in != C_out:
            self.is_identity = False
            assert C_out % 2 == 0
            self.relu = nn.ReLU(inplace=False)
            self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=1, padding=0, bias=False)
            self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=1, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(C_out, affine=affine)
        else:
            self.is_identity = True

    def forward(self, x):
        if self.is_identity:
            out = Identity(x)
        else:
            x = self.relu(x)
            out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
            out = self.bn(out)
        return out


if __name__ == '__main__':
    mixed = MixedOp(ops=[Conv2d(100, 100, 5), Conv2d(100, 32, 1), Conv2d(100, 50, 3)], choice=0, mask=[1, 1 , 1])
    num_param = sum(p.numel() for p in mixed.parameters())
    print([k for k, v in mixed.state_dict().items()])

    ex_input = torch.ones([1, 100, 32, 32])

    out = mixed(ex_input)
    print(out.shape)
    print(num_param)
