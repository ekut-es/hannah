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
import logging

import dgl
import numpy as np
import torch
from hydra.utils import instantiate
from tabulate import tabulate

from hannah.callbacks.summaries import MacSummaryCallback
from hannah.nas.graph_conversion import GraphConversionTracer, model_to_graph

logger = logging.getLogger(__name__)


class BackendPredictor:
    """A predictor class that instantiates the model and uses the backends predict function to predict performance metrics"""

    def predict(self, config):
        backend = instantiate(config.backend)
        model = instantiate(
            config.module,
            dataset=config.dataset,
            model=config.model,
            optimizer=config.optimizer,
            features=config.features,
            scheduler=config.get("scheduler", None),
            normalizer=config.get("normalizer", None),
            _recursive_=False,
        )

        model.setup("train")
        metrics = backend.estimate(model)

        logger.info("Predicted performance metrics")
        for k in metrics.keys():
            logger.info("%s: %s", k, metrics[k])

        return metrics


class MACPredictor:
    """A predictor class that instantiates the model and calculates abstract metrics"""

    def predict(self, config):
        model = instantiate(
            config.module,
            dataset=config.dataset,
            model=config.model,
            optimizer=config.optimizer,
            features=config.features,
            scheduler=config.get("scheduler", None),
            normalizer=config.get("normalizer", None),
            _recursive_=False,
        )

        model.setup("train")

        predictor = MacSummaryCallback()

        metrics = predictor.predict(model)

        logger.info("Predicted performance metrics")
        for k in metrics.keys():
            logger.info("%s: %s", k, metrics[k])

        return metrics


class GCNPredictor:
    """A predictor class that instantiates the model and uses the backends predict function to predict performance metrics"""

    def __init__(self, model):
        self.predictor = instantiate(model)

    def predict(self, config):
        backend = instantiate(config.backend)
        model = instantiate(
            config.module,
            dataset=config.dataset,
            model=config.model,
            optimizer=config.optimizer,
            features=config.features,
            scheduler=config.get("scheduler", None),
            normalizer=config.get("normalizer", None),
            _recursive_=False,
        )

        model.setup("train")

        nx_graph = model_to_graph(model)
        dgl_graph = to_dgl_graph(nx_graph)

        result, std_dev = self.predictor.predict(dgl_graph)

        print(result, std_dev)

        metrics = {}

        logger.info("Predicted performance metrics")
        for k in metrics.keys():
            logger.info("%s: %s", k, metrics[k])

        return metrics


def to_dgl_graph(nx_graph):

    node_num = {}
    fea_tensor = []
    fea_len = 0
    for num, n in enumerate(nx_graph.nodes):
        feature_vec = torch.tensor(nx_graph.nodes[n]["features"])
        fea_len = max(len(feature_vec), fea_len)
        fea_tensor.append(feature_vec)
        node_num[n] = num

    print("input feature len: ", fea_len)

    padded_features = []
    for vec in fea_tensor:
        padded_vec = np.pad(
            vec, (0, fea_len - len(vec)), mode="constant", constant_values=0
        )
        padded_vec = torch.tensor(padded_vec)
        padded_features.append(padded_vec)

    src = []
    dst = []
    for i, j in nx_graph.edges:
        src.append(node_num[i])
        dst.append(node_num[j])

    fea_tensor = torch.vstack(padded_features)
    g = dgl.graph(data=(src, dst))
    g.ndata["features"] = fea_tensor
    g = dgl.add_self_loop(g)

    return g
