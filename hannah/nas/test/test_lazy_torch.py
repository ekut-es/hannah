from hannah.nas.core.parametrized import is_parametrized
from hannah.nas.parameters.lazy import Lazy
from hannah.nas.parameters.parametrize import parametrize
from hannah.nas.parameters.parameters import IntScalarParameter

import torch.nn as nn
import torch

conv2d = Lazy(nn.Conv2d)
relu = Lazy(nn.ReLU)
linear = Lazy(nn.Linear)


@parametrize
class LazyTorchModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv0 = conv2d(in_channels=3,
                            out_channels=self.add_param('conv0_out_channels', IntScalarParameter(4, 128 , 4)),
                            kernel_size=self.add_param('conv0_kernel_size', IntScalarParameter(1, 7, 2)),
                            padding='same')

        self.relu = nn.ReLU()
        self.conv1 = conv2d(in_channels=self._PARAMETERS['conv0_out_channels'],
                            out_channels=self.add_param('conv1_out_channels', IntScalarParameter(4, 128 , 4)),
                            kernel_size=self.add_param('conv1_kernel_size', IntScalarParameter(1, 7, 2)),
                            padding='same')

        self.conv2 = conv2d(in_channels=self._PARAMETERS['conv1_out_channels'],
                            out_channels=self.add_param('conv2_out_channels', IntScalarParameter(4, 128 , 4)),
                            kernel_size=self.add_param('conv2_kernel_size', IntScalarParameter(1, 7, 2)),
                            padding='same')

        self.linear = linear(self._PARAMETERS['conv1_out_channels'] * 16 * 16, 10)

        self.cond(self._PARAMETERS['conv1_out_channels'] > 64)

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


if __name__ == '__main__':
    mod = LazyTorchModule()
    x = torch.ones(1, 3, 16, 16)
    out = mod(x)
    print(mod.parameters())
    print(mod._conditions)
    print(out)
