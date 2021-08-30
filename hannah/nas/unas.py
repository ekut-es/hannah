import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any, Dict
import omegaconf

import torch

from hydra.utils import instantiate
from joblib import Parallel, delayed

from omegaconf import OmegaConf
import numpy as np
from pytorch_lightning import LightningModule

from .aging_evolution import AgingEvolution
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything, reset_seed
from ..callbacks.optimization import HydraOptCallback
from ..utils import common_callbacks, clear_outputs, fullname

msglogger = logging.getLogger("nas")


@dataclass
class WorklistItem:
    parameters: Any
    results: Dict[str, float]


def run_training(num, config):
    os.makedirs(str(num), exist_ok=True)
    os.chdir(str(num))
    config = OmegaConf.create(config)
    logger = TensorBoardLogger(".")

    seed = config.get("seed", 1234)
    if isinstance(seed, list) or isinstance(seed, omegaconf.ListConfig):
        seed = seed[0]
    seed_everything(seed, workers=True)

    if config.trainer.gpus is not None:
        if isinstance(config.trainer.gpus, int):
            num_gpus = config.trainer.gpus
            gpu = num % num_gpus
        elif len(config.trainer.gpus) == 0:
            num_gpus = torch.cuda.device_count()
            gpu = num % num_gpus
        else:
            gpu = config.trainer.gpus[num % len(config.trainer.gpus)]

        if gpu >= torch.cuda.device_count():
            logger.warning(
                "GPU %d is not available on this device using GPU %d instead",
                gpu,
                gpu % torch.cuda.device_count(),
            )
            gpu = gpu % torch.cuda.device_count()

        config.trainer.gpus = [gpu]

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

    return opt_callback.result(dict=True)


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
        # Presample Population

        # Sample Population

        pass


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

        self.predictor = None
        if predictor is not None:
            self.predictor = instantiate(predictor, _recursive_=False)

        self.worklist = []
        self.presample = presample

    def _sample(self):
        parameters = self.optimizer.next_parameters()
        config = OmegaConf.merge(self.config, parameters.flatten())

        if self.predictor:
            predicted_metrics = self.predictor.predict(config)

            satisfied_bounds = []
            for k, v in predicted_metrics.items():
                if k in self.bounds:
                    distance = v / self.bounds[k]
                    msglogger.info(f"{k}: {float(v):.8f} ({float(distance):.2f})")
                    satisfied_bounds.append(distance <= 1.2)

            worklist_item = WorklistItem(parameters, predicted_metrics)

            if all(satisfied_bounds):
                self.worklist.append(worklist_item)
        else:
            worklist_item = WorklistItem(parameters, {})
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
                            num, OmegaConf.to_container(config, resolve=True)
                        )
                        for num, config in enumerate(configs)
                    ]
                )
                for result, item in zip(results, self.worklist):
                    parameters = item.parameters
                    metrics = {**item.results, **result}
                    for k, v in metrics.items():
                        metrics[k] = float(v)

                    self.optimizer.tell_result(parameters, metrics)
