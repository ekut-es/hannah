import torch
import torch.nn as nn
from hannah.models.capsule_net.expressions import padding_expression
from hannah.models.capsule_net.utils import handle_parameter
from hannah.nas.expressions.shapes import conv2d_shape, identity_shape, linear_shape, pool_shape
from hannah.nas.parameters.lazy import Lazy
from hannah.nas.parameters.parametrize import parametrize
from hannah.nas.expressions.arithmetic import Ceil
from hannah.nas.expressions.choice import Choice
from hannah.nas.expressions.metrics import conv2d_macs, conv2d_weights, linear_macs, linear_weights


conv2d = Lazy(nn.Conv2d, shape_func=conv2d_shape)
linear = Lazy(nn.Linear, shape_func=linear_shape)
batch_norm = Lazy(nn.BatchNorm2d, shape_func=identity_shape)
relu = Lazy(nn.ReLU)
tensor = Lazy(torch.Tensor, shape_func=identity_shape)
identity = Lazy(nn.Identity)
avg_pooling = Lazy(nn.AvgPool2d, shape_func=pool_shape)
max_pooling = Lazy(nn.MaxPool2d, shape_func=pool_shape)



@parametrize
class Relu(nn.Module):
    def __init__(self, id) -> None:
        super().__init__()
        self.id = id
        self.act = relu(self.id + '.relu')

    def initialize(self):
        self.act_mod = self.act.instantiate()

    def forward(self, x):
        return self.act_mod(x)


@parametrize
class Identity(nn.Module):
    def __init__(self, id) -> None:
        super().__init__()
        self.id = id
        self.act = identity(self.id + '.identity')

    def initialize(self):
        self.act_mod = self.act.instantiate()

    def forward(self, x):
        return self.act_mod(x)


@parametrize
class BatchNorm(nn.Module):
    def __init__(self, params, id, inputs) -> None:
        super().__init__()
        self.params = params
        self.id = id
        self.inputs = inputs
        self.input_shape = inputs[0]
        self.channels = self.input_shape[1]

        self.bn = batch_norm(self.id + '.batch_norm', num_features=self.channels, inputs=inputs)

    def initialize(self):
        self.tbn = self.bn.instantiate()

    def forward(self, x):
        return self.tbn(x)


@parametrize
class Subsampling(nn.Module):
    def __init__(self) -> None:
        super().__init__()


# FIXME: Expand padding calculations to allow other windows and strides
@parametrize
class Pooling(nn.Module):
    def __init__(self, params, id, inputs) -> None:
        super().__init__()
        self.params = params
        self.id = id
        self.inputs = inputs

        self.kernel_size = handle_parameter(self, params.kernel_size, f"{self.id}.kernel_size")
        self.stride = handle_parameter(self, params.stride, f"{self.id}.stride")

        self.avg = avg_pooling(self.id + ".avg_pool",
                               inputs=inputs,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=padding_expression(self.kernel_size, self.stride))

        self.max = max_pooling(self.id + ".max_pool",
                               inputs=inputs,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=padding_expression(self.kernel_size, self.stride))

        self.mods = {'max': self.max,
                     'avg': self.avg}
        self.choice = handle_parameter(self, params.type, f"{self.id}.type")
        self.active_module = Choice(self.mods, self.choice)

    def initialize(self):
        self.tpool = self.active_module.evaluate().instantiate()

    def forward(self, x):
        return self.tpool(x)

    @property
    def shape(self):
        return [self.active_module.get('shape')[0],
                self.active_module.get('shape')[1],
                self.active_module.get('shape')[2],
                self.active_module.get('shape')[3]]

    @property
    def macs(self):
        return 0 # FIXME:

    @property
    def weights(self):
        return 0




@parametrize
class Convolution(nn.Module):
    def __init__(self, params, id, inputs) -> None:
        super().__init__()
        self.id = id
        self.inputs = inputs
        input_shape = inputs[0]
        self.in_channels = input_shape[1]
        self.out_channels = handle_parameter(self, params.out_channels, f"{self.id}.out_channels")
        self.kernel_size = handle_parameter(self, params.kernel_size, name=f"{self.id}.kernel_size")
        self.stride = handle_parameter(self, params.stride, f"{self.id}.stride")

        self.conv = conv2d(self.id + ".conv",
                           inputs=inputs,
                           in_channels=self.in_channels,
                           out_channels=self.out_channels,
                           kernel_size=self.kernel_size,
                           stride=self.stride,
                           padding=padding_expression(self.kernel_size, self.stride))

    def initialize(self):
        self.tconv = self.conv.instantiate()

    def forward(self, x):
        out = self.tconv(x)
        return out

    @property
    def shape(self):
        return self.conv.shape

    @property
    def macs(self):
        return conv2d_macs(self.inputs[0], self.shape, self.conv.kwargs)

    @property
    def weights(self):
        return conv2d_weights(self.inputs[0], self.shape, self.conv.kwargs)


@parametrize
class DepthwiseConvolution(nn.Module):
    def __init__(self, params, id, inputs) -> None:
        super().__init__()
        self.params = params
        self.id = id
        self.inputs = inputs

        input_shape = inputs[0]
        self.in_channels = input_shape[1]
        self.kernel_size = handle_parameter(self, params.kernel_size, f"{self.id}.kernel_size")
        self.stride = handle_parameter(self, params.stride, f"{self.id}.stride")

        self.conv = conv2d(self.id + ".depthwise",
                           inputs=inputs,
                           in_channels=self.in_channels,
                           out_channels=self.in_channels,
                           kernel_size=self.kernel_size,
                           stride=self.stride,
                           groups=self.in_channels,
                           padding=padding_expression(self.kernel_size, self.stride))

    def initialize(self):
        self.tconv = self.conv.instantiate()

    def forward(self, x):
        return self.tconv(x)

    @property
    def shape(self):
        return self.conv.shape

    @property
    def macs(self):
        return conv2d_macs(self.inputs[0], self.shape, self.conv.kwargs)

    @property
    def weights(self):
        return conv2d_weights(self.inputs[0], self.shape, self.conv.kwargs)


@parametrize
class PointwiseConvolution(nn.Module):
    def __init__(self, out_channels, id, inputs) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.id = id
        self.inputs = inputs
        input_shape = inputs[0]
        self.in_channels = input_shape[1]

        self.conv = conv2d(self.id + ".pointwise",
                           inputs=inputs,
                           in_channels=self.in_channels,
                           out_channels=self.out_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

    def initialize(self):
        self.tconv = self.conv.instantiate()

    def forward(self, x):
        return self.tconv(x)

    @property
    def shape(self):
        return self.conv.shape

    @property
    def macs(self):
        return conv2d_macs(self.inputs[0], self.shape, self.conv.kwargs)

    @property
    def weights(self):
        return conv2d_weights(self.inputs[0], self.shape, self.conv.kwargs)


@parametrize
class Linear(nn.Module):
    def __init__(self, params, input, labels) -> None:
        super().__init__()
        self.params = params
        self.input = input
        self.labels = labels

        in_features = self.input[1] * self.input[2] * self.input[3]
        self._linear = self.add_param('linear',
                                      linear("linear",
                                             inputs=[self.input],
                                             in_features=in_features,
                                             out_features=self.labels))

    def initialize(self):
        self.linear = self._linear.instantiate()

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.linear(out)
        return out

    @property
    def shape(self):
        return self._linear.shape

    @property
    def shape(self):
        return self._linear.shape

    @property
    def macs(self):
        return linear_macs(self.input, self.shape, self._linear.kwargs)

    @property
    def weights(self):
        return linear_weights(self.input, self.shape, self._linear.kwargs)
