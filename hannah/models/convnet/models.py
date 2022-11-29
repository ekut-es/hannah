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

# class Exit:
#     def __init__(self, layers, choice) -> None:
#         self.layers = layers
#         self.choice = choice



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
    def __init__(self, input_shape, id, depth) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.depth = self.add_param(f'{id}.depth', depth)
        self.modules = []
        self.relu = nn.ReLU()
        self.id = id
        self.depth = depth

        strides = []

        previous = input_shape
        for d in RangeIterator(self.depth, instance=False):
            index = d - 1
            in_channels = self.input_shape[1] if d-1 == 0 else self._PARAMETERS[f'{self.id}.conv{index-1}.out_channels']
            out_channels = self.add_param(f'{self.id}.conv{index}.out_channels', IntScalarParameter(4, 128 , 4))
            kernel_size = self.add_param(f'{self.id}.conv{index}.kernel_size', CategoricalParameter([1, 3, 5, 7]))
            stride = self.add_param(f'{self.id}.conv{index}.stride', CategoricalParameter([1, 2]))

            strides.append(stride)

            layer = conv2d(f'{self.id}.conv{index}',
                           inputs=[previous],
                           in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding_expression(kernel_size, stride))
            self.modules.append(layer)
            previous = layer

        self.cond(stride_product(strides) <= self.input_shape[2])

    def exmple_input(self):
        return torch.ones(self.input_shape)

    def initialize(self):
        self.torch_modules = torch.nn.ModuleDict()
        for d in RangeIterator(self.depth, instance=False):
            self.torch_modules[self.modules[d-1].id.replace(".", "_")] = self.modules[d-1].instantiate()

    def forward(self, x):
        out = x
        for d in RangeIterator(self.depth, instance=True):
            out = self.modules[d-1].instantiate()(out)
            out = self.relu(out)
        return out


@parametrize
class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = (1, 3, 32, 32)  # FIXME:
        self.depth = IntScalarParameter(1, 3)
        self.conv_block = self.add_param("convs", ConvReluBlock(self.input_shape, 'convs', self.depth))

        # FIXME: Dynamically define last layer
        # Choice expression?
        last = Choice(self.conv_block.modules, self.depth - 1)
        in_features = last.get('kwargs')['out_channels'] * last.get('shape')[2] * last.get('shape')[3]

        # Alternatively to the following, one can create a parametrized class "Classifier" which
        # wraps the linear layer.
        self.linear = self.add_param('linear',
                                     linear("linear",
                                     inputs=[last],
                                     in_features=in_features,
                                     out_features=10))

    def example_input(self):
        return torch.ones(self.input_shape)

    def initialize(self):
        self.torch_modules = torch.nn.ModuleDict()
        self.conv_block.initialize()
        # self.torch_modules.update(self.conv_block.torch_modules)
        self.torch_modules['linear'] = self.linear.instantiate()

    def forward(self, x):
        out = self.conv_block(x)
        # print(out.shape)
        out = out.view(out.shape[0], -1)
        out = self.linear.instantiate()(out)
        return out



if __name__ == '__main__':
    net = ConvNet()
    net.parametrization()['convs']['conv0']['out_channels'].set_current(4)
    net.parametrization()['convs']['conv1']['out_channels'].set_current(8)
    net.parametrization()['convs']['conv2']['out_channels'].set_current(16)
    x = torch.ones((1, 3, 32, 32))
    net.initialize()
    out = net(x)
    print()