#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
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
import numpy as np
import torch
import torch.nn as nn

from hannah.models.ofa.submodules.elastickernelconv import ElasticConv1d


def test_elastic_conv1d_quant():
    kernel_sizes = [1, 3, 5]
    input_length = 30
    input_channels = 8
    output_channels = 8
    batch_size = 2
    dilation_sizes = [1]

    input = torch.ones((batch_size, input_channels, input_length))
    output = torch.zeros((batch_size, output_channels, input_length))

    conv = ElasticConv1d(
        input_channels,
        output_channels,
        kernel_sizes,
        dilation_sizes=dilation_sizes,
        groups=[1],
    )
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(conv.parameters(), lr=0.1)

    res = conv(input)
    orig_loss = loss_func(res, output)
    print("orig_loss:", orig_loss)

    assert res.shape == output.shape

    for i in range(5):
        optimizer.zero_grad()
        res = conv(input)
        loss = loss_func(res, output)
        loss.backward()
        optimizer.step()

    # Sample convolution size
    for i in range(20):
        kernel_size = np.random.choice(kernel_sizes)
        print("Sampled Kernel Size:", kernel_size)
        conv.set_kernel_size(kernel_size)
        optimizer.zero_grad()
        res = conv(input)
        loss = loss_func(res, output)
        loss.backward()
        optimizer.step()

    for kernel_size in kernel_sizes:
        conv.set_kernel_size(kernel_size)
        res = conv(input)
        loss = loss_func(res, output)
        print("kernel_size:", kernel_size, "loss:", loss)

        assert loss < orig_loss


if __name__ == "__main__":
    test_elastic_conv1d_quant()
