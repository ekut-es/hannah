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

from hannah_optimizer.aging_evolution import AgingEvolution
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything, reset_seed
from .callbacks.optimization import HydraOptCallback
from .utils import common_callbacks, clear_outputs, fullname

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

    num_gpus = torch.cuda.device_count()
    gpu = num % num_gpus
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
        self.backend = None

        self.worklist = []
        self.presample = presample

    def _sample(self):
        parameters = self.optimizer.next_parameters()

        config = OmegaConf.merge(self.config, parameters.flatten())
        backend = instantiate(config.backend)
        model = instantiate(
            config.module,
            dataset=config.dataset,
            model=config.model,
            optimizer=config.optimizer,
            features=config.features,
            scheduler=config.get("scheduler", None),
            normalizer=config.get("normalizer", None),
        )
        model.setup("train")
        backend_metrics = backend.estimate(model)

        satisfied_bounds = []
        for k, v in backend_metrics.items():
            if k in self.bounds:
                distance = v / self.bounds[k]
                msglogger.info(f"{k}: {float(v):.8f} ({float(distance):.2f})")
                satisfied_bounds.append(distance <= 1.2)

        worklist_item = WorklistItem(parameters, backend_metrics)

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


class OFANasTrainer(NASTrainerBase):
    def __init__(
        self,
        parent_config=None,
        gpu=0,
        epochs_warmup=10,
        epochs_kernel_step=10,
        epochs_depth_step=10,
        epochs_warmup_after_width=5,
        epochs_kernel_after_width=5,
        epochs_depth_after_width=5,
        *args,
        **kwargs
    ):
        super().__init__(*args, parent_config=parent_config, **kwargs)
        # currently no backend config for OFA
        self.gpu = gpu
        self.epochs_warmup = epochs_warmup
        self.epochs_kernel_step = epochs_kernel_step
        self.epochs_depth_step = epochs_depth_step
        self.epochs_warmup_after_width = epochs_warmup_after_width
        self.epochs_kernel_after_width = epochs_kernel_after_width
        self.epochs_depth_after_width = epochs_depth_after_width

    def run(self):
        os.makedirs("ofa_nas_dir", exist_ok=True)
        os.chdir("ofa_nas_dir")
        config = OmegaConf.create(self.config)
        logger = TensorBoardLogger(".")

        seed = config.get("seed", 1234)
        if isinstance(seed, list) or isinstance(seed, omegaconf.ListConfig):
            seed = seed[0]
        seed_everything(seed, workers=True)

        config.trainer.gpus = [self.gpu]

        callbacks = common_callbacks(config)
        opt_monitor = config.get("monitor", ["val_error"])
        opt_callback = HydraOptCallback(monitor=opt_monitor)
        callbacks.append(opt_callback)
        checkpoint_callback = instantiate(config.checkpoint)
        callbacks.append(checkpoint_callback)
        trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)
        model = instantiate(
            config.module,
            dataset=config.dataset,
            model=config.model,
            optimizer=config.optimizer,
            features=config.features,
            scheduler=config.get("scheduler", None),
            normalizer=config.get("normalizer", None),
        )
        kernel_step_count = model.ofa_steps_kernel
        depth_step_count = model.ofa_steps_depth
        width_step_count = model.ofa_steps_width

        # warm-up.
        trainer.max_epochs = self.epochs_warmup
        trainer.fit(model)
        ckpt_path = "best"
        trainer.validate(ckpt_path=ckpt_path, verbose=False)
        logging.info("OFA completed warm-up.")

        # train elastic kernels
        model.progressive_shrinking_from_warmup_to_kernel()
        trainer.max_epochs = self.epochs_kernel_step
        for current_kernel_step in range(kernel_step_count):
            model.progressive_shrinking_kernel_step()
            trainer.fit(model)

        # train elastic depth
        model.progressive_shrinking_from_kernel_to_depth()
        trainer.max_epochs = self.epochs_depth_step
        for current_depth_step in range(depth_step_count):
            model.progressive_shrinking_perform_depth_step()
            trainer.fit(model)

        # TODO: eval/save for width step 0
        self.eval_model(model, trainer, 0)

        # train elastic width
        model.progressive_shrinking_from_depth_to_width()
        trainer.max_epochs = self.epochs_warmup_after_width
        for current_width_step in range(width_step_count):
            if (current_width_step == 0):
                # the very first width step (step 0) was already processed before this loop was entered.
                continue

            # re-run warmup with reduced epoch count to re-optimize with reduced width
            model.progressive_shrinking_perform_width_step()
            model.progressive_shrinking_restart_non_width()
            trainer.max_epochs = self.epochs_warmup_after_width
            trainer.fit(model)

            # re-train elastic kernels, re-optimizing after a width step with reduced epoch count
            model.progressive_shrinking_from_warmup_to_kernel()
            trainer.max_epochs = self.epochs_kernel_after_width
            for current_kernel_step in range(kernel_step_count):
                model.progressive_shrinking_kernel_step()
                trainer.fit(model)

            # re-train elastic depth, re-optimizing after a width step with reduced epoch count
            model.progressive_shrinking_from_kernel_to_depth()
            trainer.max_epochs = self.epochs_depth_after_width
            for current_depth_step in range(depth_step_count):
                model.progressive_shrinking_perform_depth_step()
                trainer.fit(model)

        # TODO: eval/save for width step n
        self.eval_model(model, trainer, current_width_step)

        # for current_depth_step in range(depth_step_count):

    # should cycle through submodels, test them, store results (under a given width step)
    def eval_model(self, model, trainer, current_depth_step):
        ckpt_path = "best"
        if trainer.fast_dev_run:
            logging.warning(
                "Trainer is in fast dev run mode, switching off loading of best model for test"
            )
            ckpt_path = None

        # reset_seed()  # run_training does this (?)
        trainer.validate(ckpt_path=ckpt_path, verbose=False)
