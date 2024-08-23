import torch
from hannah.nas.functional_operators.executor import BasicExecutor
from hannah.nas.functional_operators.op import Tensor
from hannah.nas.graph_conversion import model_to_graph
from hannah.nas.performance_prediction.features.dataset import OnlineNASGraphDataset, get_features, to_dgl_graph
from hannah.models.embedded_vision_net.models import embedded_vision_net, search_space
import pandas as pd


def test_online_dataset():
    input = Tensor(name='input', shape=(1, 3, 32, 32), axis=("N", "C", "H", "W"))
    space = embedded_vision_net("test", input, 10, 128)
    space.sample()
    model = BasicExecutor(space)
    model.initialize()

    x = torch.ones(input.shape())
    nx_graph = model_to_graph(model, x)
    fea = get_features(nx_graph)
    
    fea = fea.astype('float32')
    for i, n in enumerate(nx_graph.nodes):
        nx_graph.nodes[n]['features'] = fea.iloc[i].to_numpy()
    dgl_graph = to_dgl_graph(nx_graph)

    graphs = [dgl_graph]
    labels = [1.0]

    dataset = OnlineNASGraphDataset(graphs, labels)


if __name__ == '__main__':
    test_online_dataset()