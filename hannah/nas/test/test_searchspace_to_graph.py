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
import pytest
import torch


from hannah.models.ai8x.models import ai8x_search_space
from hannah.models.embedded_vision_net.models import embedded_vision_net
from hannah.models.conv_vit.models import conv_vit
from hannah.nas.functional_operators.op import Tensor
from hannah.nas.graph_conversion import model_to_graph
from hannah.nas.functional_operators.executor import BasicExecutor

from hannah.nas.performance_prediction.features.dataset import (
    get_features,
    to_dgl_graph,
)

from hannah.nas.parameters.parametrize import set_parametrization
from hannah.nas.search.sampler.random_sampler import RandomSampler
from hannah.nas.constraints.random_walk import RandomWalkConstraintSolver
from flaky import flaky


@flaky(max_runs=5)
@pytest.mark.parametrize("model", [ai8x_search_space, embedded_vision_net])
def test_model_conversion(model):
    input = Tensor(name="input", shape=(1, 3, 32, 32), axis=("N", "C", "H", "W"))
    model = model("test", input, 10)
    model.sample()

    print("Init sampler")
    sampler = RandomSampler(None, model.parametrization(flatten=True))

    print("Init solver")
    solver = RandomWalkConstraintSolver()

    sampled_params, keys = sampler.next_parameters()

    solver.solve(model, sampled_params)

    parameters = solver.get_constrained_params(sampled_params)

    set_parametrization(parameters, model.parametrization(flatten=True))

    executor = BasicExecutor(model)
    executor.initialize()

    x = torch.ones(input.shape())

    nx_graph = model_to_graph(executor, x)
    fea = get_features(nx_graph)
    for i, n in enumerate(nx_graph.nodes):
        nx_graph.nodes[n]["features"] = fea.iloc[i].to_numpy()
    dgl_graph = to_dgl_graph(nx_graph)


if __name__ == "__main__":
    test_model_conversion(ai8x_search_space)
