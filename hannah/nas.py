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
        # logger = TensorBoardLogger(".")

        seed = config.get("seed", 1234)
        if isinstance(seed, list) or isinstance(seed, omegaconf.ListConfig):
            seed = seed[0]
        seed_everything(seed, workers=True)

        # TODO: Select GPU if available
        config.trainer.gpus = None

        callbacks = common_callbacks(config)
        opt_monitor = config.get("monitor", ["val_error"])
        opt_callback = HydraOptCallback(monitor=opt_monitor)
        callbacks.append(opt_callback)
        checkpoint_callback = instantiate(config.checkpoint)
        callbacks.append(checkpoint_callback)
        self.config = config
        # trainer will be initialized by rebuild_trainer
        self.trainer = None
        model = instantiate(
            config.module,
            dataset=config.dataset,
            model=config.model,
            optimizer=config.optimizer,
            features=config.features,
            scheduler=config.get("scheduler", None),
            normalizer=config.get("normalizer", None),
            _recursive_=False
        )
        model.setup("fit")
        ofa_model = model.model
        self.kernel_step_count = ofa_model.ofa_steps_kernel
        self.depth_step_count = ofa_model.ofa_steps_depth
        self.width_step_count = ofa_model.ofa_steps_width

        self.submodel_metrics = {}

        # warm-up.
        self.rebuild_trainer("warmup", self.epochs_warmup)
        self.trainer.fit(model)
        ckpt_path = "best"
        self.trainer.validate(ckpt_path=ckpt_path, verbose=False)
        logging.info("OFA completed warm-up.")

        # train elastic kernels
        for current_kernel_step in range(self.kernel_step_count):
            if (current_kernel_step == 0):
                # step 0 is the full model, and was processed during warm-up
                continue
            # add a kernel step
            ofa_model.progressive_shrinking_add_kernel()
            self.rebuild_trainer(f"kernel_{current_kernel_step}", self.epochs_kernel_step)
            self.trainer.fit(model)
        logging.info("OFA completed kernel matrices.")

        # train elastic depth
        for current_depth_step in range(self.depth_step_count):
            if (current_depth_step == 0):
                # step 0 is the full model, and was processed during warm-up
                continue
            # add a depth reduction step
            ofa_model.progressive_shrinking_add_depth()
            self.rebuild_trainer(f"depth_{current_depth_step}", self.epochs_depth_step)
            self.trainer.fit(model)
        logging.info("OFA completed depth steps.")

        self.eval_model(model, ofa_model, 0)

        # train elastic width
        for current_width_step in range(self.width_step_count):
            if (current_width_step == 0):
                # the very first width step (step 0) was already processed before this loop was entered.
                continue

            # re-run warmup with reduced epoch count to re-optimize with reduced width
            ofa_model.progressive_shrinking_perform_width_step()
            # lock sampling for re-warmup
            ofa_model.progressive_shrinking_disable_sampling()
            self.rebuild_trainer(f"width_{current_width_step}_warmup", self.epochs_warmup_after_width)
            self.trainer.fit(model)

            # re-train elastic kernels, re-optimizing after a width step with reduced epoch count
            for current_kernel_step in range(self.kernel_step_count):
                if (current_kernel_step == 0):
                    # step 0 is the full model, and was processed during warm-up
                    continue
                # re-add kernel steps
                ofa_model.progressive_shrinking_add_kernel()
                self.rebuild_trainer(f"width_{current_width_step}_kernel_{current_kernel_step}", self.epochs_kernel_after_width)
                self.trainer.fit(model)

            # re-train elastic depth, re-optimizing after a width step with reduced epoch count
            for current_depth_step in range(self.depth_step_count):
                if (current_depth_step == 0):
                    # step 0 is the full model, and was processed during warm-up
                    continue
                # re-add depth steps
                ofa_model.progressive_shrinking_add_depth()
                self.rebuild_trainer(f"width_{current_width_step}_depth_{current_depth_step}", self.epochs_depth_after_width)
                self.trainer.fit(model)

            logging.info(f"OFA completed re-training for width step {current_width_step}.")

            self.eval_model(model, ofa_model, current_width_step)

        print(self.submodel_metrics)
        np.save("ofa_submodel_metrics_dict.npy", self.submodel_metrics)

    # cycle through submodels, test them, store results (under a given width step)
    def eval_model(self, lightning_model, model, current_width_step):
        # disable sampling in forward during evaluation.
        model.eval_mode = True
        self.submodel_metrics[current_width_step] = {}
        # reset_seed()  # run_training does this (?)
        # reset target values to step through
        model.reset_all_kernel_sizes()
        for current_kernel_step in range(self.kernel_step_count):
            self.submodel_metrics[current_width_step][current_kernel_step] = {}
            if (current_kernel_step > 0):
                # iteration 0 is the full model with no stepping
                model.step_down_all_kernels()

            model.reset_active_depth()
            for current_depth_step in range(self.depth_step_count):
                if (current_depth_step > 0):
                    # iteration 0 is the full model with no stepping
                    model.active_depth -= 1

                # extracted_model = model.extract_module_from_depth_step(current_depth_step)
                self.rebuild_trainer(f"Eval K {current_kernel_step}, D {current_depth_step}, W {current_width_step}")
                logging.info(
                    f"OFA validating Kernel {current_kernel_step}, Depth {current_depth_step}, Width {current_width_step}"
                )
                validation_results = self.trainer.validate(lightning_model, ckpt_path=None, verbose=True)
                self.submodel_metrics[current_width_step][current_kernel_step][current_depth_step] = validation_results[0]
                # print(validation_results)

        # revert to normal operation after eval.
        model.eval_mode = False

    def rebuild_trainer(self, step_name: str, epochs: int = 1):
        logger = TensorBoardLogger(".", version=step_name)
        callbacks = common_callbacks(self.config)
        self.trainer = instantiate(
            self.config.trainer,
            callbacks=callbacks,
            logger=logger,
            max_epochs=epochs
        )
