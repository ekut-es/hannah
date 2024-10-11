from hannah.models.embedded_vision_net.expressions import Tensor
from hannah.nas.functional_operators.op import scope, search_space
from hannah.nas.functional_operators.operators import AdaptiveAvgPooling, BatchNorm, Conv2d, Linear, Relu
from hannah.nas.parameters import CategoricalParameter, IntScalarParameter


def conv2d(input, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, padding=None):
    in_channels = input.shape()[1]
    weight = Tensor(name='weight',
                    shape=(out_channels, in_channels, kernel_size, kernel_size),
                    axis=('O', 'I', 'kH', 'kW'),
                    grad=True)

    conv = Conv2d(stride=stride, dilation=dilation, groups=groups, padding=padding)(input, weight)
    return conv


def linear(input, out_features):
    input_shape = input.shape()
    in_features = input_shape[1] * input_shape[2] * input_shape[3]
    weight = Tensor(name='weight',
                    shape=(in_features, out_features),
                    axis=('in_features', 'out_features'),
                    grad=True)

    out = Linear()(input, weight)
    return out


@scope
def batch_norm(input):
    # https://stackoverflow.com/questions/44887446/pytorch-nn-functional-batch-norm-for-2d-input
    n_chans = input.shape()[1]
    running_mu = Tensor(name='running_mean', shape=(n_chans,), axis=('c',))
    running_std = Tensor(name='running_std', shape=(n_chans,), axis=('c',))
    # running_mu.data = torch.zeros(n_chans)  # zeros are fine for first training iter
    # running_std = torch.ones(n_chans)  # ones are fine for first training iter
    return BatchNorm()(input, running_mu, running_std)


def relu(input):
    return Relu()(input)


def adaptive_avg_pooling(input):
    return AdaptiveAvgPooling()(input)


def conv_bn_relu(input, out_channels, kernel_size, stride):
    out = conv2d(input, out_channels=out_channels, stride=stride, kernel_size=kernel_size)
    out = batch_norm(out)
    out = relu(out)
    return out


@search_space
def convnet(name, input, num_classes):
    out_channels = IntScalarParameter(
        16, 128, step_size=8, name="out_channels"
    )
    kernel_size = CategoricalParameter([3, 5, 7, 9], name="kernel_size")
    stride = CategoricalParameter([1, 2], name="stride")

    net = conv_bn_relu(input, out_channels=out_channels.new(), kernel_size=kernel_size.new(), stride=stride.new())
    net = conv_bn_relu(net, out_channels=out_channels.new(), kernel_size=kernel_size.new(), stride=stride.new())
    net = adaptive_avg_pooling(net)
    net = linear(net, num_classes)
    return net
