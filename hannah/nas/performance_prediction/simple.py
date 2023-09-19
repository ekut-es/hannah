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
import pandas as pd
import torch
import json
import networkx as nx
import tqdm
from hydra.utils import instantiate
from tabulate import tabulate

from pathlib import Path

from hannah.callbacks.summaries import FxMACSummaryCallback, MacSummaryCallback
from hannah.nas.graph_conversion import GraphConversionTracer, model_to_graph
from hannah.nas.performance_prediction.features.dataset import OnlineNASGraphDataset, get_features, to_dgl_graph
from hannah.nas.performance_prediction.gcn.predictor import prepare_dataloader

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
    def __init__(self, predictor='default') -> None:
        self._predictor = predictor

    def predict(self, model, input = None):
        if self._predictor == 'fx':
            predictor = FxMACSummaryCallback()
        else:
            predictor = MacSummaryCallback()

        metrics = predictor.predict(model, input=input)

        logger.info("Predicted performance metrics")
        for k in metrics.keys():
            logger.info("%s: %s", k, metrics[k])

        return metrics


class GCNPredictor:
    """A predictor class that instantiates the model and uses the backends predict function to predict performance metrics"""

    def __init__(self, model):
        self.predictor = instantiate(model)
        self.graphs = []
        self.labels = []


    def load(self, result_folder : str):
        result_folder = Path(result_folder)
        for i, data_path in tqdm.tqdm(enumerate(result_folder.glob("model_*.json"))):
            #if i % 500 == 0:
            i#print("Processing graph {}".format(i))

            d = json.load(data_path.open())

            nx_graph = nx.json_graph.node_link_graph(d["graph"])

            #FIXME: make features configurable
            result = d["metrics"]["val_error"]

            fea = get_features(nx_graph)

            for i, n in enumerate(nx_graph.nodes):
                nx_graph.nodes[n]['features'] = fea.iloc[i].to_numpy()

            dgl_graph = to_dgl_graph(nx_graph)

            self.graphs.append(dgl_graph)
            self.labels.append(result)

        self.train()

    def predict(self, model, input):
        model = model.model  # FIXME: Decide when to use pl_module and when to use model

        model.train()

        nx_graph = model_to_graph(model, input)
        fea = get_features(nx_graph)
        for i, n in enumerate(nx_graph.nodes):
            nx_graph.nodes[n]['features'] = fea.iloc[i].to_numpy()
        dgl_graph = to_dgl_graph(nx_graph)

        result, std_dev = self.predictor.predict(dgl_graph)

        print(result, std_dev)

        metrics = {'val_error': result}

        logger.info("Predicted performance metrics")
        for k in metrics.keys():
            logger.info("%s: %s", k, metrics[k])

        return metrics

    def update(self, new_data, input):
        for item, result in new_data:
            nx_graph = model_to_graph(item.model, input)
            fea = get_features(nx_graph)

            for i, n in enumerate(nx_graph.nodes):
                nx_graph.nodes[n]['features'] = fea.iloc[i].to_numpy()
            dgl_graph = to_dgl_graph(nx_graph)
            self.graphs.append(dgl_graph)
            self.labels.append(result)
        self.train()

    def train(self):
        dataset = OnlineNASGraphDataset(self.graphs, self.labels)
        train_dataloader, test_dataloader = prepare_dataloader(dataset, batch_size=32, train_test_split=1)
        self.predictor.train_and_fit(train_dataloader, num_epochs=20, verbose=25)
