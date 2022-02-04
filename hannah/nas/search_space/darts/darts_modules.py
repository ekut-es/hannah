import torch.nn as nn
import torch.nn.functional as F
import torch
import traceback

# Taken from DARTS :
# https://github.com/quark0/darts/blob/master/cnn/operations.py


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True
    ):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class MixedOp(nn.Module):
    def __init__(self, choice, stride, in_channels, out_channels) -> None:
        super().__init__()
        if choice == 0:
            self.op = (
                Identity()
                if stride == 1
                else FactorizedReduce(in_channels, out_channels)
            )
        elif choice == 1:
            self.op = Zero(stride=stride)
        elif choice == 2:
            self.op = nn.MaxPool2d(3, stride, padding=1)
        elif choice == 3:
            self.op = nn.AvgPool2d(3, stride, padding=1)
        elif choice == 4:
            self.op = SepConv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                affine=False,
            )
        elif choice == 5:
            self.op = SepConv(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=stride,
                padding=2,
                affine=False,
            )
        elif choice == 6:
            self.op = DilConv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=2,
                dilation=2,
                affine=False,
            )
        elif choice == 7:
            self.op = DilConv(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=stride,
                padding=4,
                dilation=2,
                affine=False,
            )
        else:
            raise Exception("Invalid choice")

    def forward(self, *seq):
        out = self.op(seq[0])
        return out


class MixedOpWS(nn.Module):
    def __init__(self, alphas, stride, in_channels, out_channels) -> None:
        super().__init__()
        self.alphas = nn.Parameter(torch.from_numpy(alphas))
        self.ops = nn.ModuleList()
        self.ops.append(
            Identity() if stride == 1 else FactorizedReduce(in_channels, out_channels)
        )
        self.ops.append(Zero(stride=stride))
        self.ops.append(nn.MaxPool2d(3, stride, padding=1))
        self.ops.append(nn.AvgPool2d(3, stride, padding=1))
        self.ops.append(
            SepConv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                affine=False,
            )
        )
        self.ops.append(
            SepConv(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=stride,
                padding=2,
                affine=False,
            )
        )
        self.ops.append(
            DilConv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=2,
                dilation=2,
                affine=False,
            )
        )
        self.ops.append(
            DilConv(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=stride,
                padding=4,
                dilation=2,
                affine=False,
            )
        )
        assert len(self.ops) == len(self.alphas)

    def set_alphas(self, alphas):
        self.alphas = nn.Parameter(torch.from_numpy(alphas))

    def forward(self, *seq):
        softmaxed_alphas = F.softmax(self.alphas, dim=0)
        out = None
        try:
            for i, op in enumerate(self.ops):
                if out is None:
                    out = softmaxed_alphas[i] * op(seq[0])
                else:
                    out += softmaxed_alphas[i] * op(seq[0])
        except Exception as e:
            print(str(e))
            print(traceback.format_exc())

            print("Error")
        return out


class Classifier(nn.Module):
    def __init__(self, C, num_classes) -> None:
        super().__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C, num_classes)

    def forward(self, *seq):
        out = self.global_pooling(seq[0])
        out = self.linear(out.view(out.size(0), -1))
        return out


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(
                5, stride=3, padding=0, count_include_pad=False
            ),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


# From naslib https://github.com/automl/NASLib
class Stem(nn.Module):
    """
    This is used as an initial layer directly after the
    image input.
    """

    def __init__(self, C_out):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, C_out, 3, padding=1, bias=False), nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        out = self.seq(x)
        return out


class Input(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        # print('Input: {} {} {}'.format(self.in_channels, self.out_channels, self.stride))

        if self.in_channels != self.out_channels and stride == 1:
            self.relu = nn.ReLU(inplace=False)
            self.conv = nn.Conv2d(
                self.in_channels, self.out_channels, 1, stride=1, padding=0, bias=False
            )
            self.bn = nn.BatchNorm2d(self.out_channels, affine=True)
        elif stride == 2:
            assert self.out_channels % 2 == 0
            self.relu = nn.ReLU(inplace=False)
            self.conv_1 = nn.Conv2d(
                self.in_channels,
                self.out_channels // 2,
                1,
                stride=stride,
                padding=0,
                bias=False,
            )
            self.conv_2 = nn.Conv2d(
                self.in_channels,
                self.out_channels // 2,
                1,
                stride=stride,
                padding=0,
                bias=False,
            )
            self.bn = nn.BatchNorm2d(self.out_channels, affine=True)

    def forward(self, *seq):
        if self.stride == 2:
            x = self.relu(seq[0])
            if x.shape[2] % 2 == 1 or x.shape[3] % 2 == 1:
                x = F.pad(x, (0, x.shape[3] % 2, 0, x.shape[2] % 2), "constant", 0)
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
