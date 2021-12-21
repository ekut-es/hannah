import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import hannah.models.factory.qconfig as qc


class Model(nn.Module):
    def __init__(self, g, input_shape) -> None:
        super().__init__()
        self.layers = nn.ModuleDict()
        self.g = g
        g.infer_shapes(input_shape)
        for node in g:
            self.layers[self.g.nodes[node]['id']] = node.to_torch(len(input_shape[2:]), self.g)

    def forward(self, x):
        def _compute(node, input):
            in_edges = self.g.in_edges(node)
            if len(in_edges) > 0:
                args = []
                for u, v in in_edges:
                    args.append(_compute(u, input))
                if len(args) == 1:
                    args = args[0]
                try:
                    out = self.layers[self.g.nodes[node]['id']](args)
                except Exception as e:
                    print(str(e))
                    print(self.g.nodes[node]['id'])

            else:
                out = self.layers[self.g.nodes[node]['id']](input)

            # print("node: {} out: {}".format(node, out.shape))
            return out

        last = list(self.g.nodes)[-1]
        out = _compute(last, x)
        return out


# Wrapper for torch.add
class Add(nn.Module):
    def __init__(self, in_channels, out_channel, dim=0) -> None:
        super(Add, self).__init__()
    def __call__(self, args):
        # maybe do padding and stuff
        if len(args) == 2:
            return torch.add(*args)
        elif len(args) > 2:
            return torch.sum(torch.stack(args), dim=0)
        else:
            return args

    def forward(self, *seq):
        return torch.add(seq)

    def __repr__(self) -> str:
        return 'torch.add()'


class Concat(nn.Module):
    def __init__(self, dim=0) -> None:
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *seq):
        return torch.cat(seq, dim=self.dim)

    def __call__(self, args):
        if not isinstance(args, list):
            args = [args]
        dim = np.zeros(len(args[0].shape))
        for i in range(len(dim)):
            dim[i] = max([a.shape[i] for a in args])
        if len(args) > 1:
            pass
        new_args = []
        for a in args:

            pad = np.repeat(dim[2:] - a.shape[2:], 2)
            if int(pad[0]/2) == 0:
                pad = tuple([int(pad[i]) if i % 2 == 0 else 0 for i in range(len(pad))])
            else:
                pad = tuple([int(p/2) for p in pad])
            a = nn.functional.pad(a, pad, 'constant', 0)
            new_args.append(a)
        args = new_args
        # concatenate on channel dimension
        return torch.cat(args, dim=self.dim)

    def __repr__(self) -> str:
        return 'torch.cat()'


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = ...) -> None:
        super().__init__(in_features, out_features, bias=bias)

    def __call__(self, input: Tensor) -> Tensor:
        batch_size = input.shape[0]
        input = input.view(batch_size, -1)
        return super().__call__(input)


class Quantize(nn.Module):
    def __init__(self, quant_args) -> None:
        super(Quantize, self).__init__()
        self.quantizer = qc.STEQuantize.with_args(**quant_args)()

    def __call__(self, args):
        # maybe do padding and stuff
        return self.quantizer(args)

    def forward(self, *seq):
        return self.quantizer(seq)


class Zero(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, seq):
        return torch.zeros_like(seq)


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
