from hannah.nas.expressions.types import Int
from hannah.nas.functional_operators.op import ChoiceOp, Tensor, scope
from hannah.nas.functional_operators.operators import (
    AdaptiveAvgPooling, Add, BatchNorm, Conv2d, Linear, Relu, Identity, MaxPooling, Dropout, SelfAttention2d
)
from hannah.models.embedded_vision_net.parameters import Groups


def self_attention2d(q, k, v, num_heads, d_model):
    out = SelfAttention2d(num_heads=num_heads, d_model=d_model)(q, k, v)
    return out


def conv2d(input, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, padding=None):
    in_channels = input.shape()[1]
    weight = Tensor(name='weight',
                    shape=(out_channels, in_channels, kernel_size, kernel_size),
                    axis=('O', 'I', 'kH', 'kW'),
                    grad=True)

    conv = Conv2d(stride=stride, dilation=dilation, groups=groups, padding=padding)(input, weight)
    return conv


def relu(input):
    return Relu()(input)


def conv_relu(input, out_channels, kernel_size, stride):
    out = conv2d(input, out_channels=out_channels, stride=stride, kernel_size=kernel_size)
    out = relu(out)
    return out


def grouped_conv2d(input, out_channels, kernel_size, stride, dilation=1, padding=None, groups=None):
    in_channels = input.shape()[1]
    if groups is None:
        groups = Groups(in_channels=in_channels, out_channels=out_channels, name="groups")

    grouped_channels = Int(in_channels / groups)
    weight = Tensor(name='weight',
                    shape=(out_channels, grouped_channels, kernel_size, kernel_size),
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


def add(input, other):
    return Add()(input, other)


def identity(input):
    return Identity()(input)


def adaptive_avg_pooling(input):
    return AdaptiveAvgPooling()(input)


def max_pool(input, kernel_size, stride):
    out = MaxPooling(kernel_size=kernel_size, stride=stride)(input)
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


def dropout(input, p=0.5):
    return Dropout(p=p)(input)


def choice(input, *choices, switch=None):
    return ChoiceOp(*choices, switch=switch)(input)


def dynamic_depth(*exits, switch):
    return ChoiceOp(*exits, switch=switch)()
