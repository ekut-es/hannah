from hannah.nas.functional_operators.executor import BasicExecutor
from hannah.nas.functional_operators.lazy import lazy
from hannah.nas.functional_operators.operators import Conv2d, Linear, Relu
from hannah.nas.functional_operators.op import Tensor
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter

from torch.optim import SGD
import torch.nn as nn
import torch


def conv2d(input, out_channels, kernel_size=1, stride=1, dilation=1):
    in_channels = input.shape()[1]
    weight = Tensor(name='weight',
                    shape=(out_channels, in_channels, kernel_size, kernel_size),
                    axis=('O', 'I', 'kH', 'kW'),
                    grad=True)

    conv = Conv2d(stride, dilation)(input, weight)
    return conv


def relu(input):
    return Relu()(input)


def conv_relu(input, out_channels, kernel_size, stride):
    out = conv2d(input, out_channels=out_channels, stride=stride, kernel_size=kernel_size)
    out = relu(out)
    return out


def linear(input, out_features):
    input_shape = input.shape()
    in_features = input_shape[1] * input_shape[2] * input_shape[3]
    weight = Tensor(name='weight',
                    shape=(in_features, out_features),
                    axis=('in_features', 'out_features'),
                    grad=True)

    out = Linear()(input, weight)
    return out


def network(input):
    out = conv_relu(input,
                    out_channels=IntScalarParameter(32, 64, name='out_channels'),
                    kernel_size=CategoricalParameter([3, 5], name="kernel_size"),
                    stride=CategoricalParameter([2], name='stride'))
    out = conv_relu(out,
                    out_channels=IntScalarParameter(16, 32, name='out_channels'),
                    kernel_size=CategoricalParameter([3, 5], name="kernel_size"),
                    stride=CategoricalParameter([1], name='stride'))
    # out = conv_relu(out,
    #                 out_channels=IntScalarParameter(4, 64, name='out_channels'),
    #                 kernel_size=CategoricalParameter([1, 3, 5, 7], name="kernel_size"),
    #                 stride=CategoricalParameter([1, 2], name='stride'))

    out = linear(out, 10)

    return out


def test_executor():
    input = Tensor(name='input',
                   shape=(1, 3, 32, 32),
                   axis=('N', 'C', 'H', 'W'))

    net = network(input)
    executor = BasicExecutor(net)
    executor.initialize()

    optimizer = SGD(executor.parameters(), lr=0.001)
    Loss = nn.L1Loss()

    k = torch.ones(input.current_shape())
    output_shape = net.shape()
    gt = torch.ones(tuple([lazy(s) for s in output_shape]))

    pred = executor.forward(k)
    loss = Loss(gt, pred)

    loss.backward()
    optimizer.step()
    print()


if __name__ == '__main__':
    test_executor()