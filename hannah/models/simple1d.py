from typing import Any

from hannah.nas.functional_operators.operators import Relu, Conv1d, Linear, AdaptiveAvgPooling
from hannah.nas.parameters import CategoricalParameter, IntScalarParameter, parametrize
from hannah.nas.functional_operators.op import Tensor, Op, scope, ChoiceOp
from hannah.nas.functional_operators.shapes import conv_shape, padding_expression
from hannah.nas.functional_operators.lazy import lazy


import torch



def conv1d(input, out_channels, kernel_size, stride):
    in_channels = input.shape()[1]
    weight = Tensor(name='weight',
                    shape=(out_channels, in_channels, kernel_size),
                    axis=('O', 'I', 'k'),
                    grad=True)

    conv = Conv1d(stride=stride)(input, weight)
    return conv

def relu(input):
    return Relu()(input)

def adaptive_avg_pooling(input):
    return AdaptiveAvgPooling(output_size=1)(input)  

def linear(input, num_classes):
    in_features = input.shape()[1]
    weight = Tensor(name='weight',
                    shape=(in_features, num_classes),
                    axis=('I', 'O'),
                    grad=True)
    return Linear()(input, weight)

@scope
def conv_relu(input, out_channels, kernel_size, stride):
    out = conv1d(input, out_channels=out_channels, stride=stride, kernel_size=kernel_size)
    out = relu(out)
    return out

@scope
def classifier_head(input, num_classes):
    out = adaptive_avg_pooling(input)
    out = linear(out, num_classes)
    return out


def dynamic_depth(*exits, switch):
    return ChoiceOp(*exits, switch=switch)()

def space(name: str, input, num_classes: int, max_channels=512, max_depth=9):
    num_blocks = IntScalarParameter(0, max_depth, name='num_blocks')
    exits = []

    out = input

    for i in range(num_blocks.max+1):
        kernel_size = CategoricalParameter([3, 5, 7, 9], name='kernel_size')
        stride = CategoricalParameter([1, 2], name='stride')
        out_channels = IntScalarParameter(16, max_channels, step_size=8, name='out_channels')
    
        out = conv_relu(out, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        exits.append(out)

    out = dynamic_depth(*exits, switch=num_blocks)
    
    out = classifier_head(out, num_classes=num_classes)


    return out
