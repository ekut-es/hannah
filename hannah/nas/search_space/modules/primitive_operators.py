import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t, _size_any_opt_t
from typing import Union
from torch import Tensor
from hannah.nas.search_space.utils import get_same_padding


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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, mid_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = get_same_padding(kernel_size)
        if mid_channels:
            self.mid_channels = mid_channels
        else:
            self.mid_channels = in_channels

        self.depthwise = nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=self.kernel_size, padding=self.pad, stride=self.stride, groups=self.in_channels)
        self.pointwise = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

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
    """
    Factorized reduce as used in ResNet to add some sort
    of Identity connection even though the resolution does not
    match.
    If the resolution matches it resolves to identity
    """

    def __init__(self, C_in, C_out, stride=1, affine=True, **kwargs):
        super().__init__()
        self.stride = stride
        if stride == 1 and C_in == C_out:
            self.is_identity = True
        elif stride == 1:
            self.is_identity = False
            self.relu = nn.ReLU(inplace=False)
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(C_out, affine=affine)

        else:
            self.is_identity = False
            assert C_out % 2 == 0
            self.relu = nn.ReLU(inplace=False)
            self.conv_1 = nn.Conv2d(
                C_in, C_out // 2, 1, stride=stride, padding=0, bias=False
            )
            self.conv_2 = nn.Conv2d(
                C_in, C_out // 2, 1, stride=stride, padding=0, bias=False
            )
            self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        if self.is_identity:
            return x
        elif self.stride == 1:
            x = self.relu(x)
            out = self.conv(x)
            self.bn(out)
            return out
        else:
            x = self.relu(x)
            if x.shape[2] % 2 == 1 or x.shape[3] % 2 == 1:
                x = F.pad(x, (0, x.shape[3] % 2, 0, x.shape[2] % 2), 'constant', 0)
            out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
            # print('Outshape', out.shape)
            out = self.bn(out)
            return out


class LayerChoice(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x):
        out = self.layer(x)
        return out


class MergedModule(nn.Module):
    def __init__(self, module_classes: dict, **kwargs) -> None:
        super().__init__()
        self.module_classes = module_classes
        self.modules = {}
        self.name = ''
        self.params = self.restructure_kwargs(kwargs)
        for mod_name, mod_cls in self.module_classes.items():
            cls_str = str(mod_cls).split('.')[-1].split('\'')[0]
            self.name += cls_str
            self.modules[mod_name] = mod_cls(**self.params[mod_name])

    def restructure_kwargs(self, kwargs):
        args = {}
        for k, v in kwargs.items():
            mod_name, param_name = k.split('.')
            if mod_name in args:
                args[mod_name][param_name] = v
            else:
                args[mod_name] = {param_name: v}
        return args

    def forward(self, x):
        for name, mod in self.modules.items():
            out = mod(x)
        return out

    def __repr__(self):
        name_str = super().__repr__() + '(\n'
        for key, mod in self.modules.items():
            name_str += '\t({}): {}\n'.format(key, mod)
        name_str += ')'
        return name_str


if __name__ == '__main__':
    mixed = MixedOp(ops=[Conv2d(100, 100, 5), Conv2d(100, 32, 1), Conv2d(100, 50, 3)], choice=0, mask=[1, 1, 1])
    num_param = sum(p.numel() for p in mixed.parameters())
    print([k for k, v in mixed.state_dict().items()])

    ex_input = torch.ones([1, 100, 32, 32])

    out = mixed(ex_input)
    print(out.shape)
    print(num_param)
