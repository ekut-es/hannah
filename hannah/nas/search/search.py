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

import torch
import os

from abc import ABC, abstractmethod
from copy import deepcopy
from hydra.utils import instantiate, get_class
from hannah.callbacks.optimization import HydraOptCallback
from hannah.utils.utils import common_callbacks
from hannah.nas.graph_conversion import model_to_graph


class NASBase(ABC):
    def __init__(self,
                 budget=2000,
                 parent_config=None) -> None:
        self.budget = budget
        self.config = parent_config
        self.callbacks = []

    def run(self):
        self.before_search()
        self.search()
        self.after_search()

    @abstractmethod
    def before_search(self):
        pass

    @abstractmethod
    def search(self):
        pass

    @abstractmethod
    def after_search(self):
        pass


class DirectNAS(NASBase):
    def __init__(self,
                 budget=2000,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, budget=budget, **kwargs)

    def before_search(self):
        # setup logging
        self.initialize_dataset()
        self.search_space = self.build_search_space()

    def search(self):
        for i in range(self.budget):
            self.global_num = i
            self.setup_model_logging()
            parameters = self.sample()
            model = self.build_model(parameters)
            lightning_module = self.initialize_lightning_module(model)
            self.train_model(lightning_module)

            self.log_results(lightning_module)
    def after_search(self):
        self.extract_best_model()

    def build_model(self, parameters):
        # FIXME: use parameters
        model = deepcopy(self.search_space)
        model.initialize()
        return model

    def build_search_space(self):
        search_space = instantiate(self.config.model)
        return search_space


    def initialize_lightning_module(self, model):
        module = instantiate(
                self.config.module,
                model=model,
                dataset=self.config.dataset,
                optimizer=self.config.optimizer,
                features=self.config.features,
                normalizer=self.config.get("normalizer", None),
                scheduler=self.config.scheduler,
                example_input_array=self.example_input_array,
                num_classes=len(self.train_set.class_names),
                _recursive_=False,
            )
        return module

    def initialize_dataset(self):
        get_class(self.config.dataset.cls).prepare(self.config.dataset)

        # Instantiate Dataset
        train_set, val_set, test_set = get_class(self.config.dataset.cls).splits(
            self.config.dataset
        )

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.example_input_array = torch.rand([1] + train_set.size())

    def train_model(self, model):
        trainer = instantiate(self.config.trainer, callbacks=self.callbacks)
        trainer.fit(model)

    def sample(self):
        # FIXME: Decoupled sampling
        self.search_space.sample()

    def setup_model_logging(self):
        self.callbacks = common_callbacks(self.config)
        opt_monitor = self.config.get("monitor", ["val_error", "train_classifier_loss"])
        opt_callback = HydraOptCallback(monitor=opt_monitor)
        self.result_handler = opt_callback
        self.callbacks.append(opt_callback)

    def log_results(self, module):
        from networkx.readwrite import json_graph
        nx_model = model_to_graph(module.model, module.example_feature_array)
        json_data = json_graph.node_link_data(nx_model)
        if not os.path.exists("../performance_data"):
            os.mkdir("../performance_data")
        with open(f"../performance_data/model_{self.global_num}.json", "w") as res_file:
            import json

            json.dump(
                {"graph": json_data,
                 "hparams": {"batch_size": int(self.config.module.batch_size)},
                             "metrics": self.result_handler.result(dict=True),
                             "curves": self.result_handler.curves(dict=True)},
                res_file,
            )


class WeightSharingNAS(NASBase):
    def __init__(self,
                 budget=2000) -> None:
        super().__init__(budget)