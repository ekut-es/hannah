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

import copy
import logging
import os
import sys
import traceback
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

import numpy as np
import torch
from hydra.utils import get_class, instantiate
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from hannah.callbacks.optimization import HydraOptCallback
from hannah.nas.functional_operators.op import Tensor
from hannah.nas.graph_conversion import model_to_graph
from hannah.nas.performance_prediction.protocol import FitablePredictor
from hannah.nas.performance_prediction.simple import MACPredictor
from hannah.nas.search.sampler.aging_evolution import FitnessFunction
from hannah.nas.search.utils import WorklistItem, save_config_to_file
from hannah.utils.utils import common_callbacks

msglogger = logging.getLogger(__name__)


class NASBase(ABC):
    def __init__(
        self,
        budget=2000,
        n_jobs=1,
        sampler=None,
        model_trainer=None,
        constraint_model=None,
        parent_config=None,
        random_state=None,
        input_shape=None,
        *args,
        **kwargs,
    ) -> None:
        self.budget = budget
        self.n_jobs = n_jobs
        self.config = parent_config
        self.callbacks = []
        self.sampler = sampler
        self.model_trainer = model_trainer
        self.predictors = {}
        self.constraint_model = constraint_model
        if random_state is None:
            self.random_state = np.random.RandomState()
        else:
            self.random_state = random_state

        self.example_input_array = None
        if input_shape is not None:
            self.example_input_array = torch.rand([1] + list(input_shape))

    def run(self):
        self.before_search()
        self.search()
        self.after_search()

    @abstractmethod
    def before_search(self): ...

    @abstractmethod
    def search(self): ...

    @abstractmethod
    def after_search(self): ...

    def add_model_trainer(self, trainer):
        self.model_trainer = trainer

    def add_sampler(self, sampler):
        self.sampler = sampler

    def get_fitness_function(self):
        # FIXME: make better configurable
        if hasattr(self, "bounds") and self.bounds is not None:
            bounds = self.bounds
            return FitnessFunction(bounds, self.random_state)
        else:
            return lambda x: x["val_error"]


class DirectNAS(NASBase):
    def __init__(
        self,
        presample=True,
        presampler=None,
        bounds=None,
        total_candidates=100,
        num_selected_candidates=10,
        constrained_sampling_on_search=False,
        constrained_init_batch=True,
        parametrization=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.presample = presample
        self.presampler = presampler
        self.bounds = bounds
        self.total_candidates = total_candidates
        self.num_selected_candidates = num_selected_candidates
        self.constrained_sampling_on_search = constrained_sampling_on_search
        self.constrained_init_batch = constrained_init_batch
        self.fixed_parametrization = parametrization

    def before_search(self):
        sys.setrecursionlimit(1500)
        self.initialize_dataset()
        self.search_space = self.build_search_space()
        parametrization = self.search_space.parametrization(flatten=True)
        if self.fixed_parametrization is not None:
            new_parametrization = {}
            for k, v in parametrization.items():
                if k in self.fixed_parametrization:
                    v.current_value = self.fixed_parametrization[k]
                else:
                    new_parametrization[k] = v
            parametrization = new_parametrization

        self.sampler = instantiate(
            self.config.nas.sampler,
            search_space=self.search_space,
            parametrization=parametrization,
            parent_config=self.config,
            _recursive_=False,
        )

        self.model_trainer = instantiate(self.config.nas.model_trainer)
        if "predictor" in self.config.nas and self.config.nas.predictor is not None:
            predictor_config = self.config.nas.predictor

            if not isinstance(predictor_config, Mapping):
                logging.warning("Predictor config is not a mapping")

                logger_config = {"default": predictor_config}

            for name, config in predictor_config.items():
                predictor = instantiate(config)
                if isinstance(predictor, FitablePredictor) and os.path.exists(
                    "performance_data"
                ):
                    predictor.load("performance_data")
                self.predictors[name] = predictor

        if self.constraint_model:
            self.constraint_model = instantiate(self.config.nas.constraint_model)

        if self.presampler:
            self.presampler = instantiate(self.config.nas.presampler)

        self.setup_model_logging()

    def init_candidates(self):
        remaining_candidates = self.total_candidates - len(self.sampler.history)
        self.candidates = []
        if remaining_candidates > 0:
            self.candidates = self.sample_candidates(
                remaining_candidates,
                remaining_candidates,
                presample=self.presample,
                constrain=self.constrained_init_batch,
            )

    def search(self):
        with Parallel(n_jobs=self.n_jobs) as executor:
            self.new_points = []

            # first batch of candidates
            self.init_candidates()

            while len(self.sampler.history) < self.budget:
                self.worklist = []

                if len(self.candidates) == 0:
                    for name, predictor in self.predictors.items():
                        try:
                            if isinstance(predictor, FitablePredictor):
                                predictor.update(
                                    self.new_points, self.example_input_array
                                )
                        except Exception as e:
                            # FIXME: Find reason for NaN in embeddings
                            msglogger.error("Updating predictor failed:")
                            msglogger.error(f"{str(e)}")

                    self.new_points = []
                    self.candidates = self.sample_candidates(
                        self.total_candidates,
                        self.num_selected_candidates,
                        presample=self.presample,
                        constrain=self.constrained_sampling_on_search,
                    )

                while len(self.worklist) < self.n_jobs and len(self.candidates) > 0:
                    try:
                        (
                            model,
                            parameters,
                            estimated_metrics,
                            satisfied_bounds,
                        ) = self.candidates.pop(0)

                        current_num = len(self.worklist)
                        task = delayed(self.model_trainer.run_training)(
                            model,
                            current_num,
                            len(self.sampler.history) + current_num,
                            self.config,
                        )

                        self.append_to_worklist(
                            parameters, task, estimated_metrics, satisfied_bounds
                        )
                    except Exception as e:
                        print(str(e))
                        print(traceback.format_exc())

                results = executor([item.task for item in self.worklist])
                for result, item in zip(results, self.worklist):
                    parameters = item.parameters
                    if self.predictors:
                        self.new_points.append(
                            (self.build_model(parameters), result["val_error"])
                        )
                    metrics = {**item.results, **result}
                    for k, v in metrics.items():
                        metrics[k] = float(v)

                    self.sampler.tell_result(parameters, metrics)

    def after_search(self):
        pass
        # self.extract_best_model()

    def sample_candidates(
        self,
        num_total,
        num_candidates=None,
        sort_key="val_error",
        presample=False,
        constrain=True,
    ):
        candidates = []
        skip_ct = 0
        while len(candidates) < num_total:
            parameters = self.sample(constrain)
            model = self.build_model(parameters)
            estimated_metrics, satisfied_bounds = self.estimate_metrics(
                copy.deepcopy(model)
            )
            if presample:
                if not self.presampler.check(model, estimated_metrics):
                    skip_ct += 1
                    continue
            ff = self.get_fitness_function()(estimated_metrics)
            estimated_metrics["ff"] = ff
            candidates.append((model, parameters, estimated_metrics, satisfied_bounds))

        if presample:
            msglogger.info(f"Skipped {skip_ct} models for not meeting constraints.")

        if "gcn" in self.predictors:
            candidates.sort(key=lambda x: x[2][sort_key])
            candidates = candidates[:num_candidates]

        # # FIXME: EXPERIMENTAL
        # candidates.sort(key=lambda x: x[2]['total_macs'], reverse=True)
        # candidates = candidates[:num_candidates]
        return candidates

    def build_model(self, parameters):
        try:
            model = self.model_trainer.build_model(self.search_space, parameters)
            module = self.initialize_lightning_module(model)
        except AssertionError as e:
            msglogger.critical(f"Instantiation failed: {e}")
        return module

    def build_search_space(self):
        input = Tensor(
            "input", shape=self.example_input_array.shape, axis=("N", "C", "H", "W")
        )
        search_space = instantiate(self.config.model, input=input, _recursive_=True)
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
            num_classes=len(self.train_set.class_names),
            _recursive_=False,
        )
        # Parameters are not part of the model but of the graph and currenlty are not automatically
        # retrievable in the module somehow
        module.param_list = torch.nn.ParameterList(list(model.parameters()))
        return module

    def initialize_dataset(self):
        get_class(self.config.dataset.cls).prepare(self.config.dataset)
        # Instantiate Dataset
        datasets = get_class(self.config.dataset.cls).splits(self.config.dataset)
        if len(datasets) == 3:
            train_set, val_set, test_set = datasets
            unlabeled_set = None
        elif len(datasets) == 4:
            train_set, unlabeled_set, val_set, test_set = datasets
        self.train_set = train_set
        self.val_set = val_set
        self.unlabeled_set = unlabeled_set
        self.test_set = test_set
        if self.example_input_array is None:
            self.example_input_array = torch.rand([1] + list(train_set.size()))

    def train_model(self, model):
        trainer = instantiate(self.config.trainer, callbacks=self.callbacks)
        trainer.fit(model)

    def sample(self, constrain=True):
        if self.constraint_model and constrain:
            while True:
                try:
                    parameters, keys = self.sampler.next_parameters()
                    if self.fixed_parametrization is not None:
                        keys.extend(list(self.fixed_parametrization.keys()))
                    # TODO: Asynchronous constraint solving
                    self.constraint_model.solve(
                        self.search_space, parameters, fix_vars=keys
                    )
                    parameters = self.constraint_model.get_constrained_params(
                        parameters
                    )
                    break
                except Exception as e:
                    print("Error occured while sampling: ")
                    print(str(e))
        #             print(traceback.format_exc())
        #             sys.exit(1)
        else:
            parameters, keys = self.sampler.next_parameters()
        return parameters

    def append_to_worklist(
        self, parameters, task, estimated_metrics={}, satisfied_bounds=[]
    ):
        worklist_item = WorklistItem(parameters, estimated_metrics, task)

        # if self.presample:
        #     if all(satisfied_bounds):
        #         self.worklist.append(worklist_item)
        # else:
        #     self.worklist.append(worklist_item)
        self.worklist.append(worklist_item)
        # FIXME: Integrate better intro current code

    def estimate_metrics(self, model):
        estimated_metrics = {}
        for name, predictor in self.predictors.items():
            metrics = predictor.predict(model, self.example_input_array)

            print(f"Predicted metrics for {name}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.8f}")

            estimated_metrics.update(metrics)

        satisfied_bounds = []
        for k, v in estimated_metrics.items():
            if k in self.bounds:
                distance = v / self.bounds[k]
                msglogger.info(f"{k}: {float(v):.8f} ({float(distance):.2f})")
                satisfied_bounds.append(distance <= 1.2)

        return estimated_metrics, satisfied_bounds

    def setup_model_logging(self):
        self.callbacks = common_callbacks(self.config)
        opt_monitor = self.config.get("monitor", ["val_error"])
        opt_callback = HydraOptCallback(monitor=opt_monitor)
        self.result_handler = opt_callback
        self.callbacks.append(opt_callback)

    def log_results(self, module):
        from networkx.readwrite import json_graph

        nx_model = model_to_graph(
            module.model, module.example_feature_array.to(module.device)
        )
        json_data = json_graph.node_link_data(nx_model)
        if not os.path.exists("../performance_data"):
            os.mkdir("../performance_data")
        with open(f"../performance_data/model_{self.global_num}.json", "w") as res_file:
            import json

            json.dump(
                {
                    "graph": json_data,
                    "hparams": {"batch_size": int(self.config.module.batch_size)},
                    "metrics": self.result_handler.result(dict=True),
                    "curves": self.result_handler.curves(dict=True),
                },
                res_file,
            )
