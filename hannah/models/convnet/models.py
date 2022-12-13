import torch
import torch.nn as nn
from hannah.nas.expressions.shapes import conv2d_shape
from hannah.nas.parameters.lazy import Lazy
from hannah.nas.parameters.parametrize import parametrize
from hannah.nas.parameters.iterators import RangeIterator
from hannah.nas.parameters.parameters import IntScalarParameter, CategoricalParameter
from hannah.nas.expressions.arithmetic import Ceil
from hannah.nas.expressions.choice import SymbolicAttr, Choice

conv2d = Lazy(nn.Conv2d, shape_func=conv2d_shape)
linear = Lazy(nn.Linear)


def padding_expression(kernel_size, stride, dilation = 1):
    """Symbolically calculate padding such that for a given kernel_size, stride and dilation
    the padding is such that the output dimension is kept the same(stride=1) or halved(stride=2).
    Note: If the input dimension is 1 and stride = 2, the calculated padding will result in
    an output with also dimension 1.

    Parameters
    ----------
    kernel_size : Union[int, Expression]
    stride : Union[int, Expression]
    dilation : Union[int, Expression], optional
        _description_, by default 1

    Returns
    -------
    Expression
    """
    p = (dilation * (kernel_size - 1) - stride + 1) / 2
    return Ceil(p)

def stride_product(expressions: list):
    res = None
    for expr in expressions:
        if res:
            res = res * expr
        else:
            res = expr
    return res


@parametrize
class ConvReluBlock(nn.Module):
    def __init__(self, params, input_shape, id, depth) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.depth = self.add_param(f'{id}.depth', depth)
        self.modules = []
        self.relus = nn.ModuleList()
        self.id = id
        self.depth = depth
        self.params = params

        strides = []

        previous = input_shape
        for d in RangeIterator(self.depth, instance=False):
            in_channels = self.input_shape[1] if d == 0 else self._PARAMETERS[f'{self.id}.conv{d-1}.out_channels']
            out_channels = self.add_param(f'{self.id}.conv{d}.out_channels', IntScalarParameter(self.params.conv.out_channels.min,
                                                                                                self.params.conv.out_channels.max,
                                                                                                self.params.conv.out_channels.step))
            kernel_size = self.add_param(f'{self.id}.conv{d}.kernel_size', CategoricalParameter(self.params.conv.kernel_size.choices))
            stride = self.add_param(f'{self.id}.conv{d}.stride', CategoricalParameter(self.params.conv.stride.choices))

            strides.append(stride)

            layer = conv2d(f'{self.id}.conv{d}',
                           inputs=[previous],
                           in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding_expression(kernel_size, stride))
            self.modules.append(layer)
            self.relus.append(nn.ReLU())
            previous = layer

        self.cond(stride_product(strides) <= self.input_shape[2])

    def initialize(self):
        self.torch_modules = nn.ModuleList()
        for d in RangeIterator(self.depth, instance=False):
            self.torch_modules.append(self.modules[d].instantiate())

    def forward(self, x):
        out = x
        for d in RangeIterator(self.depth, instance=True):
            out = self.torch_modules[d](out)
            out = self.relus[d](out)
        return out


@parametrize
class ConvNet(nn.Module):
    def __init__(self, name, params, input_shape, labels) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.labels = labels
        self.depth = IntScalarParameter(params.depth.min, params.depth.max)
        self.conv_block = self.add_param("convs", ConvReluBlock(params, self.input_shape, 'convs', self.depth))

        last = Choice(self.conv_block.modules, self.depth - 1)
        in_features = last.get('kwargs')['out_channels'] * last.get('shape')[2] * last.get('shape')[3]

        # Alternatively to the following, one can create a parametrized class "Classifier" which
        # wraps the linear layer.
        self._linear = self.add_param('linear',
                                      linear("linear",
                                      inputs=[last],
                                      in_features=in_features,
                                      out_features=self.labels))

    def initialize(self):
        self.conv_block.initialize()
        self.linear = self._linear.instantiate()

    def forward(self, x):
        out = self.conv_block(x)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

    def get_hparams(self):
        params = {}
        for key, param in self.parametrization(flatten=True).items():
                params[key] = param.current_value.item()

        return params


def create_cnn(name, input_shape, labels):
    return ConvNet(name, input_shape, labels)


if __name__ == '__main__':
    net = ConvNet()
    x = torch.randn((3, 3, 32, 32))
    net.sample()
    net.initialize()
    out = net(x)
    print()