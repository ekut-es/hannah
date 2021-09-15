import networkx as nx
import torch
from torch.nn import Module
from hannah.models.factory.qat import ConvBn1d, ConvBnReLU1d
from hannah.models.factory.qconfig import get_trax_qat_qconfig

from hannah.nas.graph_conversion import model_to_graph

from omegaconf import OmegaConf


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

    def forward(self, x):
        x = self.activation_post_process(x)
        skip = self.skip_conv(x)
        x = self.conv(x)
        x = self.conv_relu(x)
        x = x + skip
        return x


def test_graph_conversion():
    model = Model()

    graph = model_to_graph(model, torch.rand((1, 16, 50), dtype=torch.float32))

    data = nx.json_graph.node_link_data(graph)

    # graph['metrics'] = {'val_error': 0.03}

    from pprint import pprint

    pprint(data, indent=2)


if __name__ == "__main__":
    test_graph_conversion()
