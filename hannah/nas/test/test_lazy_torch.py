from hannah.nas.core.parametrized import is_parametrized
from hannah.nas.parameters.lazy import Lazy
from hannah.nas.parameters.parametrize import parametrize
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter
from hannah.nas.expressions.shapes import conv2d_shape
from hannah.nas.expressions.conditions import GECondition

import torch.nn as nn
import torch

conv2d = Lazy(nn.Conv2d, shape_func=conv2d_shape)
relu = Lazy(nn.ReLU)
linear = Lazy(nn.Linear)


class RangeIterator:
    def __init__(self, parameter, instance=False) -> None:
        self.parameter = parameter
        self.instance = instance
        self.counter = parameter.min - 1

    def __iter__(self):
        return self

    def __next__(self):
        self.counter += 1
        if self.counter <= (self.parameter.current_value if self.instance else self.parameter.max):
            return self.counter
        raise StopIteration


@parametrize
class LazyTorchModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv0 = conv2d('conv0',
                            in_channels=3,
                            out_channels=self.add_param('conv0_out_channels', IntScalarParameter(4, 128 , 4)),
                            kernel_size=self.add_param('conv0_kernel_size', IntScalarParameter(1, 7, 2)),
                            padding='same')

        symbolic_shape = self.conv0.shape((1, 3, 16, 16))

        self.relu = nn.ReLU()
        self.conv1 = conv2d('conv1',
                            in_channels=self._PARAMETERS['conv0_out_channels'],
                            out_channels=self.add_param('conv1_out_channels', IntScalarParameter(4, 128 , 4)),
                            kernel_size=self.add_param('conv1_kernel_size', IntScalarParameter(1, 7, 2)),
                            padding='same')

        symbolic_shape = self.conv1.shape(symbolic_shape)
        self.conv2 = conv2d('conv2',
                            in_channels=self._PARAMETERS['conv1_out_channels'],
                            out_channels=self.add_param('conv2_out_channels', IntScalarParameter(4, 128 , 4)),
                            kernel_size=self.add_param('conv2_kernel_size', IntScalarParameter(1, 7, 2)),
                            stride=2,
                            padding=0)

        symbolic_shape = self.conv2.shape(symbolic_shape)

        self.linear = linear(self._PARAMETERS['conv1_out_channels'] * symbolic_shape[2] * symbolic_shape[3], 10)

        self.cond(self._PARAMETERS['conv1_out_channels'] >= 64)

    def forward(self, x):
        out = self.conv0.sample()(x)
        out = self.relu(out)
        out = self.conv1.sample()(out)
        out = self.relu(out)
        out = self.conv2.sample()(out)
        out = self.relu(out)

        out = out.view(out.shape[0], -1)
        out = self.linear.sample()(out)
        return out


@parametrize
class DynamicDepthModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.depth = self.add_param('depth', IntScalarParameter(1, 5))
        self.modules = []
        self.relu = nn.ReLU()

        for d in RangeIterator(self.depth, instance=False):
            in_channels = 3 if d-1 == 0 else self._PARAMETERS[f'conv{d-2}_out_channels']
            layer = conv2d(in_channels=in_channels,
                           out_channels=self.add_param(f'conv{d-1}_out_channels', IntScalarParameter(4, 128 , 4)),
                           kernel_size=self.add_param(f'conv{d-1}_kernel_size', CategoricalParameter([1, 3, 5, 7])),
                           padding='same')
            self.modules.append(layer)

    def forward(self, x):
        out = x

        for d in RangeIterator(self.depth, instance=True):
            out = self.modules[d-1].sample()(out)
            out = self.relu(out)

        return out


def test_lazy_torch_module():
    mod = LazyTorchModule()
    for k, v in mod.parameters().items():
        v.sample()
    x = torch.ones(1, 3, 16, 16)
    out = mod(x)

    assert 'conv0_out_channels' in mod.parameters().keys()
    assert isinstance(mod._conditions[0], GECondition)
    assert out.shape == (1, 10)


def test_dynamic_depth():
    mod = DynamicDepthModule()

    x = torch.ones(1, 3, 16, 16)
    out = mod(x)
    assert out.shape == (1, 4, 16, 16)


if __name__ == '__main__':
    test_lazy_torch_module()
    test_dynamic_depth()
