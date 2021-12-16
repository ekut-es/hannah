from networkx.algorithms.tree.recognition import is_forest
import numpy as np
from copy import deepcopy
from torch import nn
import torch

import hannah.nas.space.model as model
import hannah.models.factory.qat as qat


class Operator():
    def __init__(self, op=['identity']) -> None:
        self.attrs = {}
        self.attrs['op'] = [op]
        self.instance = False

    def join(self, op):
        for key, value_list in op.attrs.items():
            if key in self.attrs:
                self.attrs[key].extend(value_list)
            else:
                self.attrs[key] = value_list

    def infer_shape(self, args):
        # For undefined op, just pass through input, i.e. return input shape
        return args[0]

    def to_torch(self, data_dim, g=None):
        return nn.Identity()

    def new(self):
        return deepcopy(self)

    def instantiate(self, cfg=None):
        self.instance = True
        if cfg:
            if isinstance(cfg, dict):
                for key, values in self.attrs.items():
                    self.attrs[key] = cfg[key]
        # else:
        #     for key, values in self.attrs.items():
        #         if isinstance(values, list):
        #             self.attrs[key] = self.attrs[key][0]
        #         else:
        #             self.attrs[key] = self.attrs[key]

    def get_knobs(self):
        knobs = {}
        for key, value in self.attrs.items():
            knobs[key] = value

        return knobs

    def __repr__(self) -> str:
        rep_str = ''
        for k, v in self.attrs.items():
            rep_str += k + ': ' + str(v) + ' '

        return rep_str


class Convolution(Operator):
    def __init__(self,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 batch_norm=False,
                 **kwargs,
                 ) -> None:

        super().__init__(op='conv')
        # impute input channel
        # self.attrs['in_channels'] = in_channels
        self.attrs['out_channels'] = out_channels
        self.attrs['kernel_size'] = kernel_size
        self.attrs['stride'] = stride
        self.attrs['padding'] = padding
        self.attrs['dilation'] = dilation
        self.attrs['groups'] = groups
        self.attrs['bias'] = bias
        self.attrs['batch_norm'] = batch_norm

        for key, value in kwargs.items():
            self.attrs[key] = value

        for k, v in self.attrs.items():
            if not isinstance(v, list):
                # TODO: Change this later to better attr-types (OneOf, ...)
                self.attrs[k] = [v]

    def copy(self):
        new_attrs = {}
        for k, v in self.attrs.items():
            new_attrs[k] = v
        op = Operator()
        op.attrs = new_attrs
        return op

    def infer_shape(self, args):
        input_shape = args[0]
        batch_size = input_shape[0]
        i_channel = input_shape[1]
        input_size = input_shape[2:]
        dims = len(input_size)
        self.attrs['in_channels'] = i_channel
        o_channel = self.attrs['out_channels']
        kernel = extend_size(self.attrs['kernel_size'], dims)
        stride = extend_size(self.attrs['stride'], dims)
        dilation = extend_size(self.attrs['dilation'], dims)

        # TODO: Support uneven padding
        if self.attrs['padding'] == 'same':
            padding = [0] * dims * 2
            for dim in range(0, dims * 2, 2):
                p = (kernel[int(dim/2)] - stride[int(dim/2)]) / 2
                assert p.is_integer()
                padding[dim] = p
                padding[dim+1] = p
        elif self.attrs['padding'] == 'half':
            self.attrs['padding'] = [0] * dims
            padding = [0] * dims * 2
            for dim in range(dims):
                assert stride[dim] == 2, 'Padding for the input to be halved can only be used when stride == 2 (not {} in dim {})'.format(stride[dim], dim)
                assert input_size[dim] % 2 == 0, 'Padding for the input to be halved only possible for input divisable by 2 (input {})'.format(input_size[dim])
                p = int(np.floor((dilation[dim] * (kernel[dim] - 1) + 1) / 2))
                padding[dim*2] = p
                padding[dim*2+1] = p
                self.attrs['padding'][dim] = p
            self.attrs['padding'] = tuple(self.attrs['padding'])
        else:
            padding = extend_size(self.attrs['padding'], dims)

        output_dims = []
        for x, k, p, s, d in zip(input_size, kernel, padding, stride, dilation):
            output_size = np.floor(((x + (2 * p) - d * (k - 1) - 1) / s) + 1)
            if not output_size.is_integer():
                raise Exception('Dimensions & parameter do not work out x: {} k: {} p: {} s: {} output: {}'.format(x, k, p, s, output_size))

            output_dims.append(int(output_size))

        return np.array([batch_size, o_channel] + output_dims)

    def get_kernel_shape(self, input):
        # assume N C H (W) input layout
        # return (O I K_h K_w)
        kernel_size = self.attrs['kernel_size']
        dims = len(input) - 2

        if isinstance(kernel_size, int) or len(kernel_size) != dims:
            kernel_size = [kernel_size]*dims
        kernel_shape = [self.attrs['out_channels'], input[1]] + kernel_size
        return kernel_shape

    def to_torch(self, data_dim, g=None):
        assert self.instance
        args = {'in_channels'  : self.attrs['in_channels'],
                'out_channels' : self.attrs['out_channels'],
                'kernel_size'     : self.attrs['kernel_size'],
                'stride'          : self.attrs['stride'],
                'padding'         : self.attrs['padding'],
                'dilation'        : self.attrs['dilation'],
                'groups'          : self.attrs['groups'],
                'bias'            : self.attrs['bias'],
                'padding_mode'    : "zeros"}

        if 'qconfig' in self.attrs:
            qconfig = self.attrs['qconfig']
            args['qconfig'] = qconfig
            args['out_quant'] = False
            if self.attrs['batch_norm']:
                if data_dim == 1:
                    conv = qat.ConvBn1d
                elif data_dim == 2:
                    conv = qat.ConvBn2d
            else:
                if data_dim == 1:
                    conv = qat.Conv1d
                elif data_dim == 2:
                    conv = qat.Conv2d

        elif data_dim == 1:
            conv = nn.Conv1d
        elif data_dim == 2:
            conv = nn.Conv2d
        elif data_dim == 3:
            conv = nn.Conv3d
        return conv(**args)

    # @classmethod
    # def from_op(cls, op):
    #     try:
    #         return cls(op.attrs['out_channels'],
    #                    op.attrs['kernel_size'],
    #                    op.attrs['stride'],
    #                    op.attrs['padding'],
    #                    op.attrs['dilation'],
    #                    op.attrs['groups'])
    #     except Exception as e:
    #         print("Failed to construct:")
    #         print(str(e))


class DepthwiseSeparableConvolution(Convolution):
    def __init__(self,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 batch_norm=False) -> None:
        super().__init__(out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, batch_norm=batch_norm)

        for k, v in self.attrs.items():
            if not isinstance(v, list):
                self.attrs[k] = [v]

    def get_kernel_shape(self, input):
        raise NotImplementedError

    def to_torch(self, data_dim, g=None):
        if data_dim == 1:
            conv = nn.Conv1d
        elif data_dim == 2:
            conv = nn.Conv2d
        elif data_dim == 3:
            conv = nn.Conv3d

        depthwise = conv(in_channels=self.attrs['in_channels'],
                         out_channels=self.attrs['in_channels'],
                         kernel_size=self.attrs['kernel_size'],
                         stride=self.attrs['stride'],
                         padding=self.attrs['padding'],  # TODO: Fix padding
                         dilation=self.attrs['dilation'],
                         groups=self.attrs['in_channels'],
                         bias=False)
        pointwise = conv(in_channels=self.attrs['in_channels'],
                         out_channels=self.attrs['out_channels'],
                         kernel_size=1,
                         padding=0,
                         bias=self.attrs['bias'])

        class depthwise_separable_conv(nn.Module):
            def __init__(self, depthwise, pointwise):
                super().__init__()
                self.depthwise = depthwise
                self.pointwise = pointwise

            def forward(self, x):
                out = self.depthwise(x)
                out = self.pointwise(out)
                return out

        return depthwise_separable_conv(depthwise, pointwise)


class Linear(Operator):
    def __init__(self, out_features) -> None:
        super().__init__(op='linear')
        self.attrs['out_features'] = out_features

        for k, v in self.attrs.items():
            if not isinstance(v, list):
                # TODO: Change this later to better attr-types (OneOf, ...)
                self.attrs[k] = [v]

    def infer_shape(self, args):
        input_shape = args[0]
        batch_size = input_shape[0]
        in_features = np.prod(input_shape[1:])
        self.attrs['in_features'] = in_features
        return np.hstack((batch_size, self.attrs['out_features']))

    def to_torch(self, data_dim, g=None):
        assert self.instance

        return model.Linear(self.attrs['in_features'], self.attrs['out_features'])


class Activation(Operator):
    def __init__(self, func) -> None:
        super().__init__(op='act')
        self.attrs['func'] = [func]

    def to_torch(self, data_dim, g=None):
        assert self.instance

        # add more activation options here
        if self.attrs['func'] == 'relu':
            return nn.ReLU()


class Relu(Activation):
    def __init__(self) -> None:
        super().__init__(func='relu')


class Combine(Operator):
    def __init__(self, mode=['add'], one_by_one_conv=[False]) -> None:
        super().__init__(op='add')
        self.attrs['mode'] = mode if isinstance(mode, list) else [mode]
        # self.attrs['one_by_one_conv'] = one_by_one_conv

    def infer_shape(self, args):
        output = None
        if self.attrs['mode'] == 'add':
            # print("Args", args)
            oc = args[0][1]
            output = np.max(args, axis=0)
        elif self.attrs['mode'] == 'concat':
            # assuming the second dim are the channel
            channel = np.sum(args[:, 1], axis=0)
            # pad the rest
            output = np.max(args, axis=0)
            output[1] = channel
        return output

    def to_torch(self, data_dim, g):
        if self.attrs['mode'] == 'add':
            in_edges = g.in_edges(self)
            out_channel = g.nodes[self]['output_shape'][1]

            if len(in_edges) > 0:
                in_channels = []
                for i, o in in_edges:
                    oc = g.nodes[i]['output_shape'][1]
                    in_channels.append(oc)
                # print('in_channels', in_channels)
                # print('out_channels', out_channel)

            return model.Add(in_channels=in_channels, out_channel=out_channel)
        elif self.attrs['mode'] == 'concat':
            return model.Concat(dim=1)


class Bias(Operator):
    def __init__(self) -> None:
        super().__init__(op='bias')


class Quantize(Operator):
    def __init__(self, quant=[None]) -> None:
        super().__init__(op='quant')
        self.attrs['quant'] = quant

        for k, v in self.attrs.items():
            if not isinstance(v, list):
                # TODO: Change this later to better attr-types (OneOf, ...)
                self.attrs[k] = [v]

    def to_torch(self, data_dim, g=None):
        assert self.instance
        if self.attrs['quant']:
            return model.Quantize(self.attrs['quant'])
        else:
            return nn.Identity()


class Dequantize(Operator):
    def __init__(self, active=[True, False]) -> None:
        super().__init__(op='dequant')
        self.attrs['active'] = active

    def to_torch(self, data_dim, g=None):
        assert self.instance
        if self.attrs['active']:
            return torch.quantization.DeQuantStub()
        else:
            return nn.Identity()


class Pooling(Operator):
    def __init__(self,
                 mode='max',
                 kernel_size=2,
                 stride=2,
                 padding=0,
                 dilation=1) -> None:
        super().__init__(op='pool')
        self.attrs['mode'] = mode
        self.attrs['kernel_size'] = kernel_size
        self.attrs['stride'] = stride
        self.attrs['padding'] = padding
        self.attrs['dilation'] = dilation

        for k, v in self.attrs.items():
            if not isinstance(v, list):
                # TODO: Change this later to better attr-types (OneOf, ...)
                self.attrs[k] = [v]

    def infer_shape(self, args):
        input_shape = args[0]
        batch_size = input_shape[0]
        i_channel = input_shape[1]
        input_size = input_shape[2:]
        dims = len(input_size)

        o_channel = i_channel
        kernel = extend_size(self.attrs['kernel_size'], dims)
        padding = extend_size(self.attrs['padding'], dims)
        stride = extend_size(self.attrs['stride'], dims)
        output_dims = []
        for x, k, p, s in zip(input_size, kernel, padding, stride):
            output_size = (x - k + 2*p) / s + 1
            if not output_size.is_integer():
                raise Exception('Invalid parameter/dimension combination (output_size {})'.format(output_size))

            output_dims.append(int(output_size))

        return np.array([batch_size, o_channel] + output_dims)

    def to_torch(self, data_dim, g=None):
        assert self.instance
        if self.attrs['mode'] == 'max':
            if data_dim == 1:
                pool = nn.MaxPool1d
            elif data_dim == 2:
                pool = nn.MaxPool2d
            elif data_dim == 3:
                pool = nn.MaxPool3d
        # TODO: Support more pool variants
        return pool(self.attrs['kernel_size'],
                    self.attrs['stride'],
                    self.attrs['padding'],
                    self.attrs['dilation'])


class Choice(Operator):
    def __init__(self, ops: list) -> None:
        super().__init__(op='choice')
        self.attrs['ops'] = ops
        # for op in ops:
        #     assert op.instance, 'Currently not supporting nested search spaces in Choice nodes'
            # self.attrs.update(op.attrs)

    def instantiate(self, cfg):
        op = self.attrs['ops'][cfg['choice']]
        self.attrs['ops'] = op
        op_id = str(op.attrs['op'][0]) + '_{}'.format(str(cfg['choice']))
        super().instantiate(cfg[op_id].update({'ops': {}}))
        self.attrs['ops'].instantiate(cfg[op_id])

    def get_knobs(self):
        knobs = {'choice': list(range(len(self.attrs['ops']))), 'ops': self.attrs['ops']}
        for i, op in enumerate(self.attrs['ops']):
            knobs[str(op.attrs['op'][0]) + '_{}'.format(i)] = op.get_knobs()
        return knobs

    def infer_shape(self, args):
        assert self.instance
        # return super().infer_shape(args)
        return self.attrs['ops'].infer_shape(args)

    def to_torch(self, data_dim, g=None):
        assert self.instance
        return self.attrs['ops'].to_torch(data_dim)


class FactorizedReduce(Operator):
    def __init__(self, out_channels, stride=[1], affine=[True]) -> None:
        super().__init__(op='factorized_reduce')
        self.attrs['out_channels'] = out_channels
        self.attrs['stride'] = stride
        self.attrs['affine'] = affine

        for k, v in self.attrs.items():
            if not isinstance(v, list):
                self.attrs[k] = [v]

    def infer_shape(self, args):
        input_shape = args[0]
        batch_size = input_shape[0]
        i_channel = input_shape[1]
        self.attrs['in_channels'] = i_channel

        input_size = input_shape[2:]
        dims = len(input_size)
        if callable(self.attrs['out_channels']):
            o_channel = self.attrs['out_channels'](i_channel)
            self.attrs['out_channels'] = o_channel
        else:
            o_channel = self.attrs['out_channels']
        stride = extend_size(self.attrs['stride'], dims)
        output_dims = []
        for x, s in zip(input_size, stride):
            output_size = np.floor(((x - 1) / s) + 1)
            if not output_size.is_integer():
                raise Exception('Invalid parameter/dimension combination (output_size {})'.format(output_size))

            output_dims.append(int(output_size))

        return np.array([batch_size, o_channel] + output_dims)

    def to_torch(self, data_dim, g=None):
        # print(self.attrs['in_channels'], self.attrs['out_channels'])
        return model.FactorizedReduce(self.attrs['in_channels'],
                                      self.attrs['out_channels'],
                                      self.attrs['stride'],
                                      self.attrs['affine'])


class Zero(Operator):
    def __init__(self) -> None:
        super().__init__(op='zero')

    def to_torch(self, data_dim, g=None):
        return model.Zero()


def extend_size(value, dim):
    if not isinstance(value, list):
        value = [value]
    elif len(value) > 1 and len(value) != dim:
        raise Exception('Wrong parameter tuple size for the given dimensions')
    if len(value) != dim:
        value = value * dim
    return value


def get_random_cfg(operator: Operator):
    knobs = operator.get_knobs()
    cfg = {}
    for k, v in knobs.items():
        i = np.random.choice(range(len(v)))
        cfg[k] = v[i]
    return cfg
