from hannah.nas.core.parametrized import is_parametrized
from hannah.nas.parameters.lazy import Lazy
from hannah.nas.parameters.parametrize import parametrize
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter
from hannah.nas.parameters.iterators import RangeIterator
from hannah.nas.expressions.shapes import conv2d_shape
from hannah.nas.expressions.conditions import GECondition

import torch.nn as nn
import torch

conv2d = Lazy(nn.Conv2d, shape_func=conv2d_shape)
relu = Lazy(nn.ReLU)
linear = Lazy(nn.Linear)


@parametrize
class LazyTorchModule(nn.Module):
    def __init__(self, input_shape) -> None:
        super().__init__()

        self.conv0 = conv2d('conv0',
                            inputs=[input_shape],
                            in_channels=3,
                            out_channels=self.add_param('conv0_out_channels', IntScalarParameter(4, 128 , 4)),
                            kernel_size=self.add_param('conv0_kernel_size', IntScalarParameter(1, 7, 2)),
                            padding='same')

        self.relu0 = relu("relu0", inputs=[self.conv0])
        self.conv1 = conv2d('conv1',
                            inputs=[self.relu0],
                            in_channels=self._PARAMETERS['conv0_out_channels'],
                            out_channels=self.add_param('conv1_out_channels', IntScalarParameter(4, 128 , 4)),
                            kernel_size=self.add_param('conv1_kernel_size', IntScalarParameter(1, 7, 2)),
                            padding='same')

        self.relu1 = relu("relu1", inputs=[self.conv1])
        self.conv2 = conv2d('conv2',
                            inputs=[self.relu1],
                            in_channels=self._PARAMETERS['conv1_out_channels'],
                            out_channels=self.add_param('conv2_out_channels', IntScalarParameter(4, 128 , 4)),
                            kernel_size=self.add_param('conv2_kernel_size', IntScalarParameter(1, 7, 2)),
                            stride=2,
                            padding=0)

        self.relu2 = relu("relu2", inputs=[self.conv2])
        self.linear = linear("linear",
                             inputs=[self.relu2],
                             in_features=self._PARAMETERS['conv2_out_channels'] * self.relu2.shape[2] * self.relu2.shape[3],
                             out_features=10)

        self.cond(self._PARAMETERS['conv1_out_channels'] >= 64)

    def initialize(self):
        self.torch_modules = nn.ModuleList()
        self.torch_modules.append(self.conv0.instantiate())
        self.torch_modules.append(self.relu0.instantiate())
        self.torch_modules.append(self.conv1.instantiate())
        self.torch_modules.append(self.relu1.instantiate())
        self.torch_modules.append(self.conv2.instantiate())
        self.torch_modules.append(self.relu2.instantiate())


    def forward(self, x):
        out = x
        for mod in self.torch_modules:
            out = mod(out)
        out = out.view(out.shape[0], -1)
        out = self.linear.instantiate()(out)
        return out


@parametrize
class DynamicDepthModule(nn.Module):
    def __init__(self, input_shape, id, depth) -> None:
        super().__init__()
        self.depth = self.add_param(f'{id}.depth', depth)
        self.modules = []
        self.relu = nn.ReLU()
        self.id = id
        self.depth = depth

        previous = input_shape
        for d in RangeIterator(self.depth, instance=False):
            in_channels = 3 if d == 0 else self._PARAMETERS[f'{self.id}.conv{d-1}.out_channels']

            layer = conv2d(f'{self.id}.conv{d}',
                           inputs=[previous],
                           in_channels=in_channels,
                           out_channels=self.add_param(f'{self.id}.conv{d}.out_channels', IntScalarParameter(4, 128 , 4)),
                           kernel_size=self.add_param(f'{self.id}.conv{d}.kernel_size', CategoricalParameter([1, 3, 5, 7])),
                           padding='same')
            self.modules.append(layer)
            previous = layer

    def initialize(self):
        self.torch_modules = nn.ModuleList()
        for d in RangeIterator(self.depth, instance=False):
            self.torch_modules.append(self.modules[d].instantiate())

    def forward(self, x):
        out = x
        for d in RangeIterator(self.depth, instance=True):
            out = self.torch_modules[d].to(x)(out)
            out = self.relu(out)
        return out


@parametrize
class BranchyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_shape = (1, 3, 16, 16)

        # self.add_param necessary for the modules to be retrievable by .parameters()
        self.main_branch = self.add_param("block", DynamicDepthModule(input_shape, 'block', depth=IntScalarParameter(1, 5)))
        self.residual = self.add_param("residual", DynamicDepthModule(input_shape, 'residual', depth=IntScalarParameter(1, 1)))

    def initialize(self):
        self.main_branch.initialize()
        self.residual.initialize()

    def forward(self, x):
        out = self.main_branch(x)
        res = self.residual(x)
        out = torch.add(out, res)
        return out


def test_lazy_torch_module():
    input_shape = (1, 3, 16, 16)
    mod = LazyTorchModule(input_shape=input_shape)

    for k, v in mod.parametrization().items():
        v.sample()

    x = torch.ones(*input_shape)

    mod.initialize()
    out = mod(x)

    assert 'conv0_out_channels' in mod.parametrization().keys()
    assert isinstance(mod._conditions[0], GECondition)
    assert out.shape == (1, 10)


def test_dynamic_depth():
    mod = DynamicDepthModule((1, 3, 16, 16), "block", IntScalarParameter(1, 5))

    x = torch.ones(1, 3, 16, 16)
    mod.initialize()
    out = mod(x)
    assert out.shape == (1, 4, 16, 16)


def test_branchy():
    mod = BranchyModule()

    x = torch.ones(1, 3, 16, 16)
    mod.initialize()
    out = mod(x)
    assert out.shape == (1, 4, 16, 16)


if __name__ == '__main__':
    test_lazy_torch_module()
    test_dynamic_depth()
    test_branchy()
