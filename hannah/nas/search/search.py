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
from joblib import Parallel, delayed
import numpy as np
from omegaconf import OmegaConf
import torch
import os

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from hydra.utils import instantiate, get_class
import yaml
from hannah.callbacks.optimization import HydraOptCallback
from hannah.nas.search.model_trainer.simple_model_trainer import SimpleModelTrainer
from hannah.nas.search.sampler.aging_evolution import AgingEvolution
from hannah.nas.search.utils import WorklistItem, save_config_to_file
from hannah.utils.utils import common_callbacks
from hannah.nas.graph_conversion import model_to_graph

msglogger = logging.getLogger(__name__)

class NASBase(ABC):
    def __init__(self,
                 budget=2000,
                 sampler=None,
                 model_trainer=None,
                 predictor=None,
                 parent_config=None) -> None:
        self.budget = budget
        self.config = parent_config
        self.callbacks = []
        self.sampler = sampler
        self.model_trainer = model_trainer
        self.predictor = predictor

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

    def add_model_trainer(self, trainer):
        self.model_trainer = trainer

    def add_sampler(self, sampler):
        self.sampler = sampler


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
        parameters = self.search_space.parametrization(flatten=True)
        self.sampler = instantiate(self.config.nas.sampler, parameters=parameters)
        self.model_trainer = instantiate(self.config.nas.model_trainer)

    def search(self):
        for i in range(self.budget):
            self.global_num = i
            self.setup_model_logging()
            parameters = self.sample()
            model = self.model_trainer.build_model(self.search_space, self.config, parameters)
            lightning_module = self.initialize_lightning_module(model)
            self.train_model(lightning_module)

            self.log_results(lightning_module)

    def after_search(self):
        self.extract_best_model()

    def build_model(self, parameters):
        try:
            model = self.model_trainer.build_model(self.search_space, parameters)
            module = self.initialize_lightning_module(model)
        except AssertionError as e:
                 msglogger.critical(f"Instantiation failed: {e}")
        return module

    def build_search_space(self):
        search_space = instantiate(self.config.model, _recursive_=True)
        return search_space

    # FIXME: Fully move to model trainer?
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
        train_set, val_set, test_set = get_class(self.config.dataset.cls).splits(self.config.dataset)
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.example_input_array = torch.rand([1] + train_set.size())

    def train_model(self, model):
        trainer = instantiate(self.config.trainer, callbacks=self.callbacks)
        trainer.fit(model)

    def sample(self):
        parameters = self.sampler.sample()
        return parameters

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

class AgingEvolutionNAS(DirectNAS):
    def __init__(self,
                 budget=2000,
                 parametrization=None,
                 bounds=None,
                 random_state=None,
                 population_size=100,
                 presample=True,
                 n_jobs=10,
                 *args, **kwargs) -> None:
        super().__init__(budget, *args, **kwargs)
        self.random_state = np.random.RandomState()
        self.parametrization = parametrization
        self.bounds = bounds
        self.population_size = population_size
        self.presample = presample
        self.worklist = []
        self.n_jobs=n_jobs

    def sample(self):
        parameters = self.sampler.next_parameters()
        return parameters

    def append_to_worklist(self, parameters, estimated_metrics={}, satisfied_bounds=[]):
        worklist_item = WorklistItem(parameters, estimated_metrics)

        if self.presample:
            if all(satisfied_bounds):
                self.worklist.append(worklist_item)
        else:
            self.worklist.append(worklist_item)

    # FIXME: Integrate better intro current code
    def estimate_metrics(self, model):
        if self.predictor:
            estimated_metrics = self.predictor.estimate(model)
        else:
            estimated_metrics = {}

        satisfied_bounds = []
        for k, v in estimated_metrics.items():
            if k in self.bounds:
                distance = v / self.bounds[k]
                msglogger.info(f"{k}: {float(v):.8f} ({float(distance):.2f})")
                satisfied_bounds.append(distance <= 1.2)

        return estimated_metrics, satisfied_bounds

    def before_search(self):
        self.initialize_dataset()
        self.search_space = self.build_search_space()
        parametrization = self.search_space.parametrization(flatten=True)
        self.sampler = AgingEvolution(parametrization=parametrization,
                                        bounds=self.bounds,
                                        population_size=self.population_size,
                                        random_state=self.random_state)

    def search(self):
        with Parallel(n_jobs=self.n_jobs) as executor:
            while len(self.sampler.history) < self.budget:
                self.worklist = []
                # Mutate current population
                models = []
                while len(self.worklist) < self.n_jobs:
                    parameters = self.sample()
                    model = self.build_model(parameters)
                    models.append(model)
                    estimated_metrics, satisfied_bounds = self.estimate_metrics(model)
                    self.append_to_worklist(parameters, estimated_metrics, satisfied_bounds)
                results = executor(
                    [
                        delayed(self.model_trainer.run_training)(
                            model,
                            num,
                            len(self.sampler.history) + num,
                            OmegaConf.to_container(self.config, resolve=True),
                        )
                        for num, model in enumerate(models)
                    ]
                )

                # save_config_to_file(self.optimizer.history, configs, results)

                for result, item in zip(results, self.worklist):
                    parameters = item.parameters
                    metrics = {**item.results, **result}
                    for k, v in metrics.items():
                        metrics[k] = float(v)

                    self.sampler.tell_result(parameters, metrics)
        print()


class WeightSharingNAS(NASBase):
    def __init__(self,
                 budget=2000) -> None:
        super().__init__(budget)