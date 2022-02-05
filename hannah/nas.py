import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import shutil
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
from .callbacks.summaries import MacSummaryCallback
from .utils import common_callbacks, clear_outputs, fullname

msglogger = logging.getLogger("nas")


@dataclass
class WorklistItem:
    parameters: Any
    results: Dict[str, float]


def run_training(num, config):
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

        num_gpus = torch.cuda.device_count()
        gpu = num % num_gpus
        config.trainer.gpus = [gpu]

        callbacks = common_callbacks(config)
        opt_monitor = config.get("monitor", ["val_error"])
        opt_callback = HydraOptCallback(monitor=opt_monitor)
        callbacks.append(opt_callback)

        checkpoint_callback = instantiate(config.checkpoint, _recursive_=False)
        callbacks.append(checkpoint_callback)
        try:
            trainer = instantiate(
                config.trainer,
                callbacks=callbacks,
                logger=logger,
                _recursive_=False,
                _convert_="partial",
            )
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

        self.worklist = []
        self.presample = presample

    def _sample(self):
        parameters = self.optimizer.next_parameters()

        config = OmegaConf.merge(self.config, parameters.flatten())

        if "backend" in config:
            estimator = instantiate(config.backend, _recursive_=False)
        else:
            estimator = MacSummaryCallback()

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
        estimated_metrics = estimator.estimate(model)

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
        epochs_width_step=10,
        elastic_kernels=False,
        elastic_depth=False,
        elastic_width=False,
        evaluate=True,
        # epochs_warmup_after_width=5,
        # epochs_kernel_after_width=5,
        # epochs_depth_after_width=5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, parent_config=parent_config, **kwargs)
        # currently no backend config for OFA
        self.epochs_warmup = epochs_warmup
        self.epochs_kernel_step = epochs_kernel_step
        self.epochs_depth_step = epochs_depth_step
        self.epochs_width_step = epochs_width_step
        self.elastic_kernels = elastic_kernels
        self.elastic_depth = elastic_depth
        self.elastic_width = elastic_width
        self.evaluate = evaluate

    def run(self):
        os.makedirs("ofa_nas_dir", exist_ok=True)
        os.chdir("ofa_nas_dir")
        config = OmegaConf.create(self.config)
        # logger = TensorBoardLogger(".")

        seed = config.get("seed", 1234)
        if isinstance(seed, list) or isinstance(seed, omegaconf.ListConfig):
            seed = seed[0]
        seed_everything(seed, workers=True)

        if not torch.cuda.is_available():
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
            _recursive_=False,
        )
        model.setup("fit")
        ofa_model = model.model
        self.kernel_step_count = ofa_model.ofa_steps_kernel
        self.depth_step_count = ofa_model.ofa_steps_depth
        self.width_step_count = ofa_model.ofa_steps_width
        ofa_model.elastic_kernels = self.elastic_kernels
        ofa_model.elastic_depth = self.elastic_depth
        ofa_model.elastic_width = self.elastic_width

        logging.info("Kernel Steps: %d", self.kernel_step_count)
        logging.info("Depth Steps: %d", self.depth_step_count)
        logging.info("Width Steps: %d", self.width_step_count)

        self.submodel_metrics_csv = (
            "width, kernel, depth, acc, total_macs, total_weights, torch_params\n"
        )
        self.random_metrics_csv = "width_steps, depth, kernel_steps, acc, total_macs, total_weights, torch_params\n"

        logging.info("Once for all Model:\n %s", str(ofa_model))

        self.warmup(model, ofa_model)

        self.train_elastic_kernel(model, ofa_model)
        self.train_elastic_depth(model, ofa_model)
        self.train_elastic_width(model, ofa_model)

        if self.evaluate:
            self.eval_model(model, ofa_model)

            # save random metrics
            print(self.random_metrics_csv)
            with open("OFA_random_sample_metrics.csv", "w") as f:
                f.write(self.random_metrics_csv)
            # save self.submodel_metrics_csv
            print(self.submodel_metrics_csv)
            with open("OFA_elastic_metrics.csv", "w") as f:
                f.write(self.submodel_metrics_csv)

    def warmup(self, model, ofa_model):
        # warm-up.
        self.rebuild_trainer("warmup", self.epochs_warmup)
        self.trainer.fit(model)
        ckpt_path = "best"
        self.trainer.validate(ckpt_path=ckpt_path, verbose=True)
        ofa_model.on_warmup_end()
        logging.info("OFA completed warm-up.")

    def train_elastic_width(self, model, ofa_model):
        if self.elastic_width:
            # train elastic width
            # first, run channel priority computation
            ofa_model.progressive_shrinking_compute_channel_priorities()

            # self.eval_model(model, ofa_model, 0)

            for current_width_step in range(self.width_step_count):
                if current_width_step == 0:
                    # the very first width step (step 0) was already processed before this loop was entered.
                    continue

                # add a width step
                ofa_model.progressive_shrinking_add_width()
                self.rebuild_trainer(
                    f"width_{current_width_step}", self.epochs_width_step
                )
                self.trainer.fit(model)
            logging.info("OFA completed width steps.")

    def train_elastic_depth(self, model, ofa_model):
        if self.elastic_depth:
            # train elastic depth
            for current_depth_step in range(self.depth_step_count):
                if current_depth_step == 0:
                    # step 0 is the full model, and was processed during warm-up
                    continue
                # add a depth reduction step
                ofa_model.progressive_shrinking_add_depth()
                self.rebuild_trainer(
                    f"depth_{current_depth_step}", self.epochs_depth_step
                )
                self.trainer.fit(model)
            logging.info("OFA completed depth steps.")

    def train_elastic_kernel(self, model, ofa_model):
        if self.elastic_kernels == True:
            # train elastic kernels
            for current_kernel_step in range(self.kernel_step_count):
                if current_kernel_step == 0:
                    # step 0 is the full model, and was processed during warm-up
                    continue
                # add a kernel step
                ofa_model.progressive_shrinking_add_kernel()
                self.rebuild_trainer(
                    f"kernel_{current_kernel_step}", self.epochs_kernel_step
                )
                self.trainer.fit(model)
            logging.info("OFA completed kernel matrices.")

    def eval_elastic_width(self, method_stack, method_index, lightning_model, model, trainer_path, loginfo_output, metrics_output):
        model.reset_all_widths()
        method = method_stack[method_index]

        for current_width_step in range(self.width_step_count):
            if current_width_step > 0:
                # iteration 0 is the full model with no stepping
                model.step_down_all_channels()

            trainer_path_tmp = trainer_path + f"W {current_width_step}, "
            loginfo_output_tmp = loginfo_output + f"Width {current_width_step}, "
            metrics_output_tmp = metrics_output + f"{current_width_step}, "

            method(method_stack, method_index + 1, lightning_model, model, trainer_path_tmp, loginfo_output_tmp, metrics_output_tmp)


    def eval_elastic_kernels(self, method_stack, method_index, lightning_model, model, trainer_path, loginfo_output, metrics_output):
        model.reset_all_kernel_sizes()
        method = method_stack[method_index]

        for current_kernel_step in range(self.kernel_step_count):
            if current_kernel_step > 0:
                # iteration 0 is the full model with no stepping
                model.step_down_all_kernels()

            trainer_path_tmp = trainer_path + f"K {current_kernel_step}, "
            loginfo_output_tmp = loginfo_output + f"Kernel {current_kernel_step}, "
            metrics_output_tmp = metrics_output + f"{current_kernel_step}, "

            method(method_stack, method_index + 1, lightning_model, model, trainer_path_tmp, loginfo_output_tmp, metrics_output_tmp)


    def eval_elatic_depth(self, method_stack, method_index, lightning_model, model, trainer_path, loginfo_output, metrics_output):
        model.reset_active_depth()
        method = method_stack[method_index]

        for current_depth_step in range(self.depth_step_count):
            if current_depth_step > 0:
                # iteration 0 is the full model with no stepping
                model.active_depth -= 1

            trainer_path_tmp = trainer_path + f"D {current_depth_step}, "
            loginfo_output_tmp = loginfo_output + f"Depth {current_depth_step}, "
            metrics_output_tmp = metrics_output + f"{current_depth_step}, "

            method(method_stack, method_index + 1, lightning_model, model, trainer_path_tmp, loginfo_output_tmp, metrics_output_tmp)


    def eval_single_model(self, method_stack, method_index, lightning_model, model, trainer_path, loginfo_output, metrics_output):
        self.rebuild_trainer(trainer_path)
        logging.info(loginfo_output)

        model.build_validation_model()
        validation_results = self.trainer.validate(
            lightning_model, ckpt_path=None, verbose=True
        )

        self.submodel_metrics_csv += metrics_output
        results = validation_results[0]
        torch_params = model.get_validation_model_weight_count()
        self.submodel_metrics_csv += f"{results['val_accuracy']}, {results['total_macs']}, {results['total_weights']}, {torch_params}"
        self.submodel_metrics_csv += "\n"

    # cycle through submodels, test them, store results (under a given width step)
    def eval_model(self, lightning_model, model):
        # disable sampling in forward during evaluation.
        model.eval_mode = True
        # reset_seed()  # run_training does this (?)
        # reset target values to step through

        eval_methods = list()

        if self.elastic_width:
            eval_methods.append(self.eval_elastic_width)

        if self.elastic_kernels:
            eval_methods.append(self.eval_elastic_kernels)

        if self.elastic_depth:
            eval_methods.append(self.eval_elatic_depth)

        if len(eval_methods) > 0:
            eval_methods.append(self.eval_single_model)
            eval_methods[0](eval_methods, 1, lightning_model, model, "Eval ", "OFA validating ", "")

        self.eval_random_combination(lightning_model, model)

        model.eval_mode = False

    def eval_random_combination(self, lightning_model, model):
        # sample a few random combinations
        prev_max_kernel = model.sampling_max_kernel_step
        prev_max_depth = model.sampling_max_depth_step
        prev_max_width = model.sampling_max_width_step
        model.sampling_max_kernel_step = model.ofa_steps_kernel - 1
        model.sampling_max_depth_step = model.ofa_steps_depth - 1
        model.sampling_max_width_step = model.ofa_steps_width - 1
        for i in range(100):
            random_state = model.sample_subnetwork()
            selected_depth = random_state["depth_step"]
            selected_kernels = random_state["kernel_steps"]
            selected_widths = random_state["width_steps"]
            selected_kernels_string = str(selected_kernels).replace(",", ";")
            selected_widths_string = str(selected_widths).replace(",", ";")
            self.rebuild_trainer(
                f"Eval random sample: D {selected_depth}, Ks {selected_kernels}, Ws {selected_widths}"
            )
            logging.info(f"OFA validating random sample:\n{random_state}")
            model.build_validation_model()
            validation_results = self.trainer.validate(
                lightning_model, ckpt_path=None, verbose=True
            )
            results = validation_results[0]
            self.random_metrics_csv += f"{selected_widths_string}, {selected_depth}, {selected_kernels_string}, "
            torch_params = model.get_validation_model_weight_count()
            self.random_metrics_csv += f"{results['val_accuracy']}, {results['total_macs']}, {results['total_weights']}, {torch_params}"
            self.random_metrics_csv += "\n"
        # revert to normal operation after eval.
        model.sampling_max_kernel_step = prev_max_kernel
        model.sampling_max_depth_step = prev_max_depth
        model.sampling_max_width_step = prev_max_width

    def rebuild_trainer(self, step_name: str, epochs: int = 1):
        logger = TensorBoardLogger(".", version=step_name)
        callbacks = common_callbacks(self.config)
        self.trainer = instantiate(
            self.config.trainer, callbacks=callbacks, logger=logger, max_epochs=epochs
        )
