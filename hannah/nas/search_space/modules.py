import torch.nn as nn
import torch.nn.functional as F
import torch


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
