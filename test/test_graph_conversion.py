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
import networkx as nx
import torch
from omegaconf import OmegaConf
from torch.nn import Module

from hannah.models.factory.pooling import ApproximateGlobalAveragePooling1D
from hannah.models.factory.qat import ConvBn1d, ConvBnReLU1d, Linear
from hannah.models.factory.qconfig import get_trax_qat_qconfig
from hannah.nas.functional_operators.op import Tensor
from hannah.nas.graph_conversion import model_to_graph
from hannah.models.convnet.models import ConvNet


class Model(Module):
    def __init__(self):
        super().__init__()
        qconfig = get_trax_qat_qconfig(
            OmegaConf.create({"bw_w": 6, "bw_b": 8, "bw_f": 8})
        )
        self.activation_post_process = qconfig.activation()
        self.conv = ConvBn1d(16, 24, 9, padding=4, qconfig=qconfig)
        self.conv_relu = ConvBnReLU1d(24, 32, 3, padding=1, stride=2, qconfig=qconfig)
        self.skip_conv = ConvBnReLU1d(16, 32, 1, stride=2, qconfig=qconfig)
        # self.pooling = ApproximateGlobalAveragePooling1D(2, qconfig=qconfig)
        self.linear = Linear(25, 12, qconfig=qconfig)

    def forward(self, x):
        x = self.activation_post_process(x)
        skip = self.skip_conv(x)
        x = self.conv(x)
        x = self.conv_relu(x)
        x = x + skip
        if hasattr(self, "pooling"):
            x = self.pooling(x)
        x = self.linear(x)
        return x


def test_graph_conversion():
    model = Model()

    test_output = model(torch.randn((1, 16, 50)))

    graph = model_to_graph(model, torch.rand((1, 16, 50), dtype=torch.float32))
    data = nx.json_graph.node_link_data(graph)

    # graph['metrics'] = {'val_error': 0.03}

    from pprint import pprint

    pprint(data, indent=2)


def test_graph_conversion_lazy_convnet():
    from omegaconf import OmegaConf

    params = {'depth': {'min': 3, 'max': 3},
              'conv': {'kernel_size': {'choices': [3, 5, 7]},
                        'stride': {'choices': [1, 2]},
                        'out_channels': {'min': 16, 'max': 64, 'step': 4}}}

    config = OmegaConf.merge(params)

    model = ConvNet(name='cnn', params=config, input_shape=[1, 3, 32, 32], labels=10)
    model.sample()
    model.initialize()
    test_output = model(torch.rand((1, 3, 32, 32), dtype=torch.float32))
    graph = model_to_graph(model, torch.rand((1, 3, 32, 32), dtype=torch.float32))


def test_graph_conversion_functional_operators():
    from hannah.models.capsule_net_v2.models import search_space
    from hannah.nas.functional_operators.executor import BasicExecutor
    import hannah.nas.functional_operators.operators

    input = Tensor(name='input',
                   shape=(1, 3, 32, 32),
                   axis=('N', 'C', 'H', 'W'))

    # space = test_net(input)

    space = search_space("net", input)
    # space.sample()
    model = BasicExecutor(space)
    model.initialize()
    x = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    # model.forward(x)
    model.find_execution_order()
    # model.create_forward()
    # out = model.forward(x)
    graph = model_to_graph(model, x)
    print()


if __name__ == "__main__":
    test_graph_conversion()
    test_graph_conversion_lazy_convnet()
    test_graph_conversion_functional_operators()
