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
import matplotlib.pyplot as plt
import networkx as nx
import pytest

from hannah.nas.ops import batched_image_tensor
from hannah.nas.parameters.parameters import IntScalarParameter
from hannah.nas.test.network import residual_block


@pytest.mark.xfail
def test_adjacency():
    input = batched_image_tensor(shape=(1, 3, 32, 32), name="input")
    graph = residual_block(
        input,
        stride=IntScalarParameter(1, 2),
        output_channel=IntScalarParameter(4, 512, 4),
    )

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


if __name__ == "__main__":
    test_adjacency()
