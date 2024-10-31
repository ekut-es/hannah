#
# Copyright (c) 2024 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from hannah.nas.functional_operators.op import Tensor, scope, search_space
from hannah.nas.functional_operators.operators import (
    AdaptiveAvgPooling,
    BatchNorm,
    Conv2d,
    Linear,
    Permute,
    Relu,
    Reshape,
)
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter


@scope
def spatial_to_channel(input, grid_size):
    batch, in_channels, H, W = input.shape()

    input.cond(in_channels == 1)
    input.cond(H % grid_size == 0)
    input.cond(W % grid_size == 0)

    patch_size = grid_size * grid_size

    permuted = Permute([0, 2, 3, 1])(input)
    reshaped = Reshape(shape=(batch, H // grid_size, W // grid_size, -1))(permuted)
    out = Permute([0, 3, 1, 2])(reshaped)

    return out


def conv2d(input, out_channels, kernel_size, stride, groups=1):
    _, in_channels, _, _ = input.shape()

    weight = Tensor(
        name="weight",
        shape=(out_channels, in_channels // groups, kernel_size, kernel_size),
        axis=("O", "I", "kH", "kW"),
        grad=True,
    )
    out = Conv2d(stride=stride, padding=0, groups=groups)(input, weight)

    out.cond(in_channels % groups == 0)

    return out


@scope
def batch_norm(input):
    n_chans = input.shape()[1]
    running_mu = Tensor(name="running_mean", shape=(n_chans,), axis=("c",))
    running_std = Tensor(name="running_std", shape=(n_chans,), axis=("c",))
    return BatchNorm()(input, running_mu, running_std)


def relu(input):
    return Relu()(input)


@scope
def block(input, stride, out_channels, kernel_size, expand_ratio):
    _, in_channels, _, _ = input.shape()

    groups = in_channels * expand_ratio // 8

    out = input
    out = conv2d(out, out_channels=in_channels * expand_ratio, kernel_size=1, stride=1)
    out = batch_norm(out)
    out = relu(out)
    out = conv2d(
        out,
        out_channels=in_channels * expand_ratio,
        kernel_size=kernel_size,
        stride=stride,
        groups=groups,
    )
    out = batch_norm(out)
    out = relu(out)
    out = conv2d(out, out_channels=out_channels, kernel_size=1, stride=1)
    out = batch_norm(out)
    out = relu(out)

    return out


@scope
def backbone(input, num_classes, max_channels, max_blocks):
    out_channels = IntScalarParameter(
        16, max_channels, step_size=8, name="out_channels"
    )
    expand_ratio = IntScalarParameter(1, 4, name="expand_ratio")
    grid_size = CategoricalParameter([1, 2, 4, 8, 16], name="grid_size")

    num_blocks = IntScalarParameter(0, max_blocks, name="num_blocks")
    exits = []

    out = conv2d(
        input, out_channels=out_channels.new(), kernel_size=grid_size, stride=grid_size
    )
    out = batch_norm(out)
    for i in range(max_blocks):
        out = block(
            out,
            2,
            out_channels.new(),
            3,
            expand_ratio.new(),
        )
        exits.append(out)

    return out


def adaptive_avg_pooling(input):
    return AdaptiveAvgPooling()(input)


def linear(input, out_features):
    input_shape = input.shape()
    in_features = input_shape[1] * input_shape[2] * input_shape[3]
    weight = Tensor(
        name="weight",
        shape=(in_features, out_features),
        axis=("in_features", "out_features"),
        grad=True,
    )

    out = Linear()(input, weight)
    return out


@scope
def classifier_head(input, num_classes):
    out = input
    out = adaptive_avg_pooling(out)
    out = linear(out, num_classes)
    return out


@search_space
def medge_net(
    name,
    input,
    num_classes: int,
    max_channels=512,
    max_blocks=9,
    constraints: list[dict] = [],
):
    arch = backbone(input, num_classes, max_channels, max_blocks)
    arch = classifier_head(arch, num_classes)
    return arch
