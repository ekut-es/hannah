#
# Copyright (c) 2023 Hannah contributors.
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
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import omegaconf
import torch
import torch.package as package
import yaml
from hydra.utils import instantiate
from joblib import Parallel, delayed
from lightning.fabric.utilities.seed import reset_seed, seed_everything
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from hannah.models.convnet.models import ConvNet
from hannah.nas.core.parametrized import is_parametrized

from ...callbacks.optimization import HydraOptCallback
from ...callbacks.summaries import MacSummaryCallback
from ...utils import clear_outputs, common_callbacks, fullname
from ..graph_conversion import model_to_graph
from ..parametrization import SearchSpace
from .sampler.aging_evolution import AgingEvolutionSampler

msglogger = logging.getLogger(__name__)

import pickle
from typing import List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass()
class EvolutionResult:
    index: int
    parameters: Dict[str, Any]
    result: Dict[str, float]
    test_result: Optional[Dict[str, float]] = None
    result_curves: Optional[pd.DataFrame] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None

    def costs(self):
        return np.asarray(
            [float(self.result[k]) for k in sorted(self.result.keys())],
            dtype=np.float32,
        )


class FitnessFunction:
    def __init__(self, bounds, random_state):
        self.bounds = bounds
        self.lambdas = random_state.uniform(low=0.0, high=1.0, size=len(self.bounds))

    def __call__(self, values):
        result = 0.0
        for num, key in enumerate(self.bounds.keys()):
            if key in values:
                result += np.power(
                    self.lambdas[num] * (values[key] / self.bounds[key]), 2
                )
            else:
                logger.warning("Metric %s is missing in sample", key)
                return float("inf")
        return np.sqrt(result)


class AgingEvolution:
    """Aging Evolution based multi objective optimization"""

    def __init__(
        self,
        parametrization,
        bounds,
        population_size=100,
        sample_size=20,
        eps=0.1,
        random_state=None,
        output_folder=".",
    ):
        self.parametrization = SearchSpace(parametrization, random_state)
        self.bounds = bounds

        self.population_size = population_size
        self.sample_size = sample_size
        self.eps = eps

        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        if random_state is None:
            self.random_state = np.random.RandomState()

        self.history = []
        self.population = []
        self._pareto_points = []
        self.output_folder = Path(output_folder)
        if (self.output_folder / "history.pkl").exists():
            self.load()

    def get_fitness_function(self):
        ff = FitnessFunction(self.bounds, self.random_state)

        return ff

    def ask(self):
        return self.next_parameters()

    def next_parameters(self):
        "Returns a list of current tasks"

        parametrization = {}

        if len(self.history) < self.population_size:
            parametrization = self.parametrization.get_random()
        elif self.random_state.uniform() < self.eps:
            parametrization = self.parametrization.get_random()
        else:
            logger.info("Sampling parents")
            sample = self.random_state.choice(self.population, size=self.sample_size)
            fitness_function = self.get_fitness_function()

            sample = [x for x in sample if hasattr(x, "result")]
            fitness = [fitness_function(x.result) for x in sample]

            parent = sample[np.argmin(fitness)]

            parametrization = self.parametrization.mutate(parent.parameters)

        return parametrization

    def tell(self, parameters, metrics):
        return self.tell_result(parameters, metrics)

    def tell_result(self, parameters, metrics):
        "Tell the result of a task"

        if isinstance(metrics, dict):
            result = EvolutionResult(len(self.history), parameters, metrics)
        else:
            result = EvolutionResult(
                len(self.history),
                parameters,
                metrics.metrics,
                metrics.test_metrics,
                metrics.curves,
                metrics.start_time,
                metrics.end_time,
                metrics.duration,
            )

        self.history.append(result)
        self.population.append(result)
        if len(self.population) > self.population_size:
            self.population.pop(0)

        self.save()

        return None

    def save(self):
        history_file = self.output_folder / "history.yml"
        history_file_tmp = history_file.with_suffix(".tmp")

        with history_file_tmp.open("w") as history_data:
            yaml.dump(self.history, history_data)
        shutil.move(history_file_tmp, history_file)

    def load(self):
        # suffixes = [".pkl", ".yml"]
        suffixes = [".pkl"]
        history_file_base = self.output_folder / "history"
        for suffix in suffixes:
            history_file = history_file_base.with_suffix(suffix)
            if history_file.exists():
                break

        self.history = []
        self.population = []

        if history_file.exists():
            if suffix == ".yml":
                with history_file.open("r") as history_data:
                    self.history = yaml.unsafe_load(history_data)
            elif suffix == ".pkl":
                with history_file.open("rb") as history_data:
                    self.history = pickle.load(history_data)

        if len(self.history) > self.population_size:
            population_start = len(self.history) - self.population_size - 1
            self.population = self.history[population_start:]
        else:
            self.population = self.history

        logging.info("Loaded %d points from history", len(self.history))


@dataclass
class WorklistItem:
    parameters: Any
    results: Dict[str, float]


def run_training(
    num, global_num, config
):  # num is the number of jobs global_num is the number of models to be created
    if os.path.exists(str(num)):
        shutil.rmtree(str(num))

    os.makedirs(str(num), exist_ok=True)

    try:
        os.chdir(str(num))
        config = OmegaConf.create(config)
        logger = TensorBoardLogger(".")

        seed = config.get("seed", 1234)
        if isinstance(seed, list) or isinstance(seed, omegaconf.ListConfig):
            seed = seed[0]
        seed_everything(seed, workers=True)

        if config.trainer.devices is not None:
            if isinstance(config.trainer.devices, int):
                num_gpus = config.trainer.devices
                gpu = num % num_gpus
            elif len(config.trainer.gpus) == 0:
                num_gpus = torch.cuda.device_count()
                gpu = num % num_gpus
            else:
                gpu = config.trainer.devices[num % len(config.trainer.gpus)]

            if gpu >= torch.cuda.device_count():
                logging.warning(
                    "GPU %d is not available on this device using GPU %d instead",
                    gpu,
                    gpu % torch.cuda.device_count(),
                )
                gpu = gpu % torch.cuda.device_count()

            config.trainer.devices = [gpu]

        callbacks = common_callbacks(config)
        opt_monitor = config.get("monitor", ["val_error"])
        opt_callback = HydraOptCallback(monitor=opt_monitor)
        callbacks.append(opt_callback)

        checkpoint_callback = instantiate(config.checkpoint)
        callbacks.append(checkpoint_callback)
        try:
            trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)
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
            trainer.fit(model)
            ckpt_path = "best"
            if trainer.fast_dev_run:
                logging.warning(
                    "Trainer is in fast dev run mode, switching off loading of best model for test"
                )
                ckpt_path = None

            reset_seed()
            trainer.validate(ckpt_path=ckpt_path, verbose=False)
        except Exception as e:
            msglogger.critical("Training failed with exception")
            msglogger.critical(str(e))
            res = {}
            for monitor in opt_monitor:
                res[monitor] = float("inf")

        nx_model = model_to_graph(model.model, model.example_feature_array)
        from networkx.readwrite import json_graph

        json_data = json_graph.node_link_data(nx_model)
        if not os.path.exists("../performance_data"):
            os.makedirs("../performance_data", exist_ok=True)
        with open(f"../performance_data/model_{global_num}.json", "w") as res_file:
            import json

            json.dump(
                {"graph": json_data, "metrics": opt_callback.result(dict=True)},
                res_file,
            )

        return opt_callback.result(dict=True)
    finally:
        os.chdir("..")


class NASTrainerBase(ABC):
    def __init__(
        self,
        budget=2000,
        parent_config=None,
        parametrization=None,
        bounds=None,
        n_jobs=5,
    ):
        self.config = parent_config
        self.budget = budget
        self.parametrization = parametrization
        self.bounds = bounds
        self.n_jobs = n_jobs

    @abstractmethod
    def run(self, model):
        pass


class RandomNASTrainer(NASTrainerBase):
    def __init__(self, budget=2000, *args, **kwargs):
        super().__init__(*args, budget=budget, **kwargs)

    def fit(self, module: LightningModule):
        callbacks = common_callbacks(self.config)
        opt_monitor = self.config.get("monitor", ["val_error", "train_classifier_loss"])
        opt_callback = HydraOptCallback(monitor=opt_monitor)
        callbacks.append(opt_callback)

        trainer = instantiate(self.config.trainer, callbacks=callbacks)
        trainer.fit(module)

        from networkx.readwrite import json_graph

        nx_model = model_to_graph(module.model, module.example_feature_array)
        json_data = json_graph.node_link_data(nx_model)
        if not os.path.exists("../performance_data"):
            os.makedirs("../performance_data", exist_ok=True)
        with open(f"../performance_data/model_{self.global_num}.json", "w") as res_file:
            import json

            json.dump(
                {
                    "graph": json_data,
                    "hparams": {"batch_size": int(self.config.module.batch_size)},
                    "metrics": opt_callback.result(dict=True),
                    "curves": opt_callback.curves(dict=True),
                },
                res_file,
            )

    def run(self):
        from hydra.utils import get_class, instantiate

        # Prepare dataset
        get_class(self.config.dataset.cls).prepare(self.config.dataset)

        # Instantiate Dataset
        train_set, val_set, test_set = get_class(self.config.dataset.cls).splits(
            self.config.dataset
        )

        # self.search_space.prepare([1] + train_set.size())
        for i in range(self.budget):
            self.global_num = i
            example_input_array = torch.rand([1] + train_set.size())

            # instantiate search space
            model = instantiate(self.config.model)

            # sample from search space
            # model is still a Parametrized nn.Module with Lazy layer stubs
            # model.forward(x) will throw error
            model.sample()

            # initialize model: Lazy layers are instantiated to real torch.nn layers
            # forward works now
            model.initialize()
            module = instantiate(
                self.config.module,
                model=model,
                dataset=self.config.dataset,
                optimizer=self.config.optimizer,
                features=self.config.features,
                normalizer=self.config.get("normalizer", None),
                scheduler=self.config.scheduler,
                example_input_array=example_input_array,
                num_classes=len(train_set.class_names),
                _recursive_=False,
            )

            self.fit(module)


class AgingEvolutionNASTrainer(NASTrainerBase):
    def __init__(
        self,
        population_size=100,
        budget=2000,
        parametrization=None,
        bounds=None,
        parent_config=None,
        presample=True,
        n_jobs=10,
        predictor=None,
        seed=1234,
    ):
        super().__init__(
            budget=budget,
            bounds=bounds,
            parametrization=parametrization,
            parent_config=parent_config,
            n_jobs=n_jobs,
        )
        self.population_size = population_size

        self.random_state = np.random.RandomState()
        self.optimizer = AgingEvolution(
            parametrization=parametrization,
            bounds=bounds,
            random_state=self.random_state,
        )

        if predictor is not None:
            self.predictor = instantiate(predictor, _recursive_=False)
        else:
            self.predictor = None

        self.worklist = []
        self.presample = presample

        if self.config.get("backend", None):
            self.backend = instantiate(self.config.backend)
        else:
            self.backend = None

    # sample parameters and estimated metrics from the space
    def _sample(self):
        # sample the next parameters
        parameters = self.optimizer.next_parameters()
        config = OmegaConf.merge(self.config, parameters.flatten())

        try:
            # setup the model
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
            # train the model
            model.setup("train")

        except AssertionError as e:
            msglogger.critical(
                "Instantiation failed. Probably #input/output channels are not divisible by #groups!"
            )
            msglogger.critical(str(e))
        else:
            estimated_metrics = {}
            if self.predictor is not None:
                estimated_metrics = self.predictor.estimate(model)

            if self.backend:
                self.backend.prepare(model)
                backend_metrics = self.backend.estimate_metrics()
                estimated_metrics.update(backend_metrics)

            satisfied_bounds = []
            for k, v in estimated_metrics.items():
                if k in self.bounds:
                    distance = v / self.bounds[k]
                    msglogger.info(f"{k}: {float(v):.8f} ({float(distance):.2f})")
                    satisfied_bounds.append(distance <= 1.2)

            worklist_item = WorklistItem(parameters, estimated_metrics)

            if self.presample:
                if all(satisfied_bounds):
                    self.worklist.append(worklist_item)
            else:
                self.worklist.append(worklist_item)

    def run(self):
        with Parallel(n_jobs=self.n_jobs) as executor:
            while len(self.optimizer.history) < self.budget:
                self.worklist = []
                # Mutate current population
                while len(self.worklist) < self.n_jobs:
                    self._sample()

                # validate population
                configs = [
                    OmegaConf.merge(self.config, item.parameters.flatten())
                    for item in self.worklist
                ]

                results = executor(
                    [
                        delayed(run_training)(
                            num,
                            len(self.optimizer.history) + num,
                            OmegaConf.to_container(config, resolve=True),
                        )
                        for num, config in enumerate(configs)
                    ]
                )

                for num, (config, result) in enumerate(zip(configs, results)):
                    nas_result_path = Path("results")
                    if not nas_result_path.exists():
                        nas_result_path.mkdir(parents=True, exist_ok=True)
                    config_file_name = f"config_{len(self.optimizer.history)+num}.yaml"
                    config_path = nas_result_path / config_file_name
                    with config_path.open("w") as config_file:
                        config_file.write(OmegaConf.to_yaml(config))

                    result_path = nas_result_path / "results.yaml"
                    result_history = []
                    if result_path.exists():
                        with result_path.open("r") as result_file:
                            result_history = yaml.safe_load(result_file)
                        if not isinstance(result_history, list):
                            result_history = []

                    result_history.append(
                        {"config": str(config_file_name), "metrics": result}
                    )

                    with result_path.open("w") as result_file:
                        yaml.safe_dump(result_history, result_file)

                for result, item in zip(results, self.worklist):
                    parameters = item.parameters
                    metrics = {**item.results, **result}
                    for k, v in metrics.items():
                        metrics[k] = float(v)

                    self.optimizer.tell_result(parameters, metrics)
