from hannah.nas.ops import batched_image_tensor
from hannah.nas.test.network import residual_block
from hannah.nas.parameters.parameters import IntScalarParameter
import networkx as nx
import matplotlib.pyplot as plt


def test_adjacency():
    input = batched_image_tensor(shape=(1, 3, 32, 32), name='input')
    graph = residual_block(input, stride=IntScalarParameter(1, 2), output_channel=IntScalarParameter(4, 512, 4))

    a, indices = graph.adjacency()
    print(a)
    g = nx.from_numpy_array(a)
    nx.topological_sort(g)
    a_top = nx.to_numpy_array(g)
    print(a_top)
    # mapping = {i: n for n, i in indices.items()}
    # g = nx.relabel_nodes(g, mapping)
    # nx.draw(g, with_labels=True)
    # plt.show()

    print()


if __name__ == '__main__':
    test_adjacency()
