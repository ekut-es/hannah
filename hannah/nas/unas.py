import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any, Dict
import omegaconf

import torch
import yaml

from pathlib import Path

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

        self.random_state = np.random.RandomState(seed=seed)
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
