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

import json
import logging
from pathlib import Path

import dgl
import lightning as L
import networkx as nx
import numpy as np
import pandas as pd
import torch
import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig
from tabulate import tabulate

from hannah.backends.base import AbstractBackend
from hannah.callbacks.summaries import FxMACSummaryCallback, MacSummaryCallback
from hannah.modules.base import ClassifierModule
from hannah.nas.graph_conversion import GraphConversionTracer, model_to_graph
from hannah.nas.performance_prediction.features.dataset import (
    OnlineNASGraphDataset,
    get_features,
    to_dgl_graph,
)
from hannah.nas.performance_prediction.gcn.predictor import (
    Predictor,
    prepare_dataloader,
)

logger = logging.getLogger(__name__)


class MACPredictor:
    """A predictor class that instantiates the model and calculates abstract metrics"""

    def __init__(self, predictor="default") -> None:
        if predictor == "fx":
            self.predictor = FxMACSummaryCallback()
        else:
            self.predictor = MacSummaryCallback()

    def predict(self, model, input=None):
        metrics = self.predictor.predict(model, input=input)

        return metrics


class GCNPredictor:
    """A predictor class that instantiates the model and uses the backends predict function to predict performance metrics"""

    def __init__(self, model):
        if isinstance(model, DictConfig):
            self.predictor = instantiate(model)
        elif isinstance(model, Predictor):
            self.predictor = model
        else:
            raise Exception(
                f"type {type(model)} is not a valid type for a predictor model."
            )

        self.graphs = []
        self.labels = []

    def load(self, result_folder: str):
        result_folder = Path(result_folder)
        for i, data_path in tqdm.tqdm(enumerate(result_folder.glob("model_*.json"))):
            # if i % 500 == 0:
            i  # print("Processing graph {}".format(i))

            d = json.load(data_path.open())

            nx_graph = nx.json_graph.node_link_graph(d["graph"])

            # FIXME: make features configurable
            result = d["metrics"]["val_error"]

            fea = get_features(nx_graph)

            for i, n in enumerate(nx_graph.nodes):
                nx_graph.nodes[n]["features"] = fea.iloc[i].to_numpy()

            dgl_graph = to_dgl_graph(nx_graph)

            self.graphs.append(dgl_graph)
            self.labels.append(result)

        self.train()

    def predict(self, model, input=None):
        if input is None:
            if hasattr(model, "example_feature_array"):
                input = model.example_feature_array
            elif hasattr(model, "example_input_array"):
                input = model.example_input_array
            else:
                raise Exception("No input provided and no example input found in model")

        if isinstance(model, ClassifierModule):
            model = (
                model.model
            )  # FIXME: Decide when to use pl_module and when to use model

        model.train()

        nx_graph = model_to_graph(model, input)
        fea = get_features(nx_graph)
        for i, n in enumerate(nx_graph.nodes):
            nx_graph.nodes[n]["features"] = fea.iloc[i].to_numpy()
        dgl_graph = to_dgl_graph(nx_graph)

        result, std_dev = self.predictor.predict(dgl_graph)

        print(result, std_dev)

        metrics = {"val_error": result.item()}

        logger.info("Predicted performance metrics")
        for k in metrics.keys():
            logger.info("%s: %s", k, metrics[k])

        return metrics

    def update(self, new_data, input):
        for item, result in new_data:
            nx_graph = model_to_graph(item.model, input)
            fea = get_features(nx_graph)

            for i, n in enumerate(nx_graph.nodes):
                nx_graph.nodes[n]["features"] = fea.iloc[i].to_numpy()
            dgl_graph = to_dgl_graph(nx_graph)
            self.graphs.append(dgl_graph)
            self.labels.append(result)
        self.train()

    def train(self):
        dataset = OnlineNASGraphDataset(self.graphs, self.labels)
        train_dataloader, test_dataloader = prepare_dataloader(
            dataset, batch_size=32, train_test_split=1
        )
        self.predictor.train_and_fit(train_dataloader, num_epochs=20, verbose=25)


class BackendPredictor:
    """A predictor class that uses a backend to predict performance metrics"""

    def __init__(self, backend: AbstractBackend = None) -> None:
        self.backend = backend

    def predict(self, module: L.LightningModule, input=None):
        metrics = {}
        if self.backend:
            if module.example_input_array is None:
                module.example_input_array = (
                    input if input is not None else module.get_example_input_array()
                )
            self.backend.prepare(module)
            profile_result = self.backend.profile(input if input is not None else [])
            metrics.update(profile_result.metrics)
        return metrics
