import torch
import torch.nn as nn
from hannah.nas.expressions.shapes import conv2d_shape, identity_shape
from hannah.nas.parameters.lazy import Lazy
from hannah.nas.parameters.parametrize import parametrize
from hannah.nas.parameters.iterators import RangeIterator
from hannah.nas.parameters.parameters import IntScalarParameter, CategoricalParameter
from hannah.nas.expressions.arithmetic import Ceil
from hannah.nas.expressions.choice import SymbolicAttr, Choice
from hannah.nas.expressions.types import Int

conv2d = Lazy(nn.Conv2d, shape_func=conv2d_shape)
linear = Lazy(nn.Linear)
batch_norm = Lazy(nn.BatchNorm2d, shape_func=identity_shape)
relu = Lazy(nn.ReLU)
tensor = Lazy(torch.Tensor, shape_func=identity_shape)


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
class ConvReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, id, inputs) -> None:
        super().__init__()
        self.id = id
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = conv2d(self.id + ".conv",
                           inputs=inputs,
                           in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding_expression(kernel_size, stride))

        self.shape = self.conv.shape
        self.bn = batch_norm(self.id + ".bn", num_features=out_channels)
        self.relu = relu(self.id + ".relu")

    def initialize(self):
        self.tconv = self.conv.instantiate()
        self.tbn = self.bn.instantiate()
        self.trelu = self.relu.instantiate()

    def forward(self, x):
        out = self.tconv(x)
        out = self.tbn(out)
        out = self.trelu(out)
        return out

@parametrize
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_fmap_size, out_fmap_size, id, inputs) -> None:
        super().__init__()
        self.id = id
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_fmap = in_fmap_size
        self.out_fmap = out_fmap_size
        self.stride = Int(Ceil(in_fmap_size / out_fmap_size))
        self.conv = conv2d(id=self.id + ".residual_conv",
                           inputs=inputs,
                           in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=1,
                           stride=self.stride,
                           padding=0)
        self.activation = relu(self.id + '.relu')

    def initialize(self):
        self.tconv = self.conv.instantiate()
        self.tact = self.activation.instantiate()

    def forward(self, x):
        out = self.tconv(x)
        out = self.tact(out)
        return out


@parametrize
class ConvReluBlock(nn.Module):
    def __init__(self, params, input_shape, id, depth) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.depth = self.add_param(f'{id}.depth', depth)
        self.mods = nn.ModuleList()
        self.id = id
        self.depth = depth
        self.params = params

        self.strides = []

        previous = input_shape
        for d in RangeIterator(self.depth, instance=False):
            in_channels = self.input_shape[1] if d == 0 else previous.out_channels
            out_channels = self.add_param(f'{self.id}.conv{d}.out_channels', IntScalarParameter(self.params.conv.out_channels.min,
                                                                                                self.params.conv.out_channels.max,
                                                                                                self.params.conv.out_channels.step))
            kernel_size = self.add_param(f'{self.id}.conv{d}.kernel_size', CategoricalParameter(self.params.conv.kernel_size.choices))
            stride = self.add_param(f'{self.id}.conv{d}.stride', CategoricalParameter(self.params.conv.stride.choices))

            self.strides.append(stride)

            layer = ConvReluBn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, id=f'{self.id}.{d}', inputs=[previous])
            self.mods.append(layer)
            previous = layer

        self.last_layer = Choice(self.mods, self.depth - 1)
        self.cond(stride_product(self.strides) <= self.input_shape[2])

    def initialize(self):
        for d in RangeIterator(self.depth, instance=False):
            self.mods[d].initialize()

    def forward(self, x):
        out = x
        for d in RangeIterator(self.depth, instance=True):
            out = self.mods[d](out)
        return out

@parametrize
class ClassifierHead(nn.Module):
    def __init__(self, input, labels) -> None:
        super().__init__()
        self.labels = labels
        in_features = input.get('shape')[1] * input.get('shape')[2] * input.get('shape')[3]
        self._linear = self.add_param('linear',
                                      linear("linear",
                                      inputs=[input],
                                      in_features=in_features,
                                      out_features=self.labels))

    def initialize(self):
        self.linear = self._linear.instantiate()

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.linear(out)
        return out


@parametrize
class ResNet(nn.Module):
    def __init__(self, name, params, input_shape, labels) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.labels = labels
        self.depth = IntScalarParameter(params.depth.min, params.depth.max)
        self.num_blocks = IntScalarParameter(params.num_blocks.min, params.num_blocks.max)
        self.conv_blocks = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        next_input = self.input_shape
        for n in RangeIterator(self.num_blocks, instance=False):
            block = self.add_param(f"conv_block_{n}",
                                   ConvReluBlock(params,
                                                 next_input,
                                                 f"conv_block_{n}",
                                                 self.depth))
            # last = Choice(block.mods, self.depth - 1)
            residual_block = self.add_param(f"residual_block_{n}",
                                            ResidualBlock(next_input[1],
                                                          block.last_layer.get('shape')[1],
                                                          next_input[2],
                                                          block.last_layer.get('shape')[2],
                                                          f"residual_block_{n}",
                                                          next_input))
            next_input = [block.last_layer.get("shape")[0], block.last_layer.get("shape")[1], block.last_layer.get("shape")[2], block.last_layer.get("shape")[3]]
            self.conv_blocks.append(block)
            self.residual_blocks.append(residual_block)

        last_block = Choice(self.conv_blocks, self.num_blocks - 1)
        self.classifier = ClassifierHead(last_block.get("last_layer"), self.labels)


    def initialize(self):
        for n in RangeIterator(self.num_blocks, instance=False):
            self.conv_blocks[n].initialize()
            self.residual_blocks[n].initialize()
        self.classifier.initialize()

    def forward(self, x):
        out = x
        for n in RangeIterator(self.num_blocks, instance=True):
            block_out = self.conv_blocks[n](out)
            res_out = self.residual_blocks[n](out)
            out = torch.add(block_out, res_out)
            out = block_out

        out = self.classifier(out)
        return out

    def get_hparams(self):
        params = {}
        for key, param in self.parametrization(flatten=True).items():
                params[key] = param.current_value.item()

        return params