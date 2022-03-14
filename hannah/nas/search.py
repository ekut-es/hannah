import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import shutil
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
from ..callbacks.summaries import MacSummaryCallback
from ..utils import common_callbacks, clear_outputs, fullname

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

    finally:
        os.chdir("..")

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


class OFANasTrainer(NASTrainerBase):
    def __init__(
        self,
        parent_config=None,
        epochs_warmup=10,
        epochs_kernel_step=10,
        epochs_depth_step=10,
        epochs_width_step=10,
        epochs_dilation_step=10,
        elastic_kernels_allowed=False,
        elastic_depth_allowed=False,
        elastic_width_allowed=False,
        elastic_dilation_allowed=False,
        evaluate=True,
        random_evaluate=True,
        random_eval_number=100,
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
        self.epochs_dilation_step = epochs_dilation_step
        self.elastic_kernels_allowed = elastic_kernels_allowed
        self.elastic_depth_allowed = elastic_depth_allowed
        self.elastic_width_allowed = elastic_width_allowed
        self.elastic_dilation_allowed = elastic_dilation_allowed
        self.evaluate = evaluate
        self.random_evaluate = random_evaluate
        self.random_eval_number = random_eval_number

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
        self.dilation_step_count = ofa_model.ofa_steps_dilation
        ofa_model.elastic_kernels_allowed = self.elastic_kernels_allowed
        ofa_model.elastic_depth_allowed = self.elastic_depth_allowed
        ofa_model.elastic_width_allowed = self.elastic_width_allowed
        ofa_model.elastic_dilation_allowed = self.elastic_dilation_allowed

        logging.info("Kernel Steps: %d", self.kernel_step_count)
        logging.info("Depth Steps: %d", self.depth_step_count)
        logging.info("Width Steps: %d", self.width_step_count)

        self.submodel_metrics_csv = ""
        self.random_metrics_csv = ""

        if self.elastic_width_allowed:
            self.submodel_metrics_csv += "width, "
            self.random_metrics_csv += "width_steps, "

        if self.elastic_kernels_allowed:
            self.submodel_metrics_csv += "kernel, "
            self.random_metrics_csv += "kernel_steps, "

        if self.elastic_dilation_allowed:
            self.submodel_metrics_csv += "dilation, "
            self.random_metrics_csv += "dilation_steps, "

        if self.elastic_depth_allowed:
            self.submodel_metrics_csv += "depth, "
            self.random_metrics_csv += "depth, "

        if (
            self.elastic_width_allowed
            | self.elastic_kernels_allowed
            | self.elastic_dilation_allowed
            | self.elastic_depth_allowed
        ):
            self.submodel_metrics_csv += (
                "acc, total_macs, total_weights, torch_params\n"
            )
            self.random_metrics_csv += "acc, total_macs, total_weights, torch_params\n"

        # self.random_metrics_csv = "width_steps, depth, kernel_steps, acc, total_macs, total_weights, torch_params\n"

        logging.info("Once for all Model:\n %s", str(ofa_model))

        self.warmup(model, ofa_model)

        self.train_elastic_kernel(model, ofa_model)
        self.train_elastic_dilation(model, ofa_model)
        self.train_elastic_depth(model, ofa_model)
        self.train_elastic_width(model, ofa_model)

        if self.evaluate:
            self.eval_model(model, ofa_model)

            if self.random_evaluate:
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
        ofa_model.reset_validaton_model()
        logging.info("OFA completed warm-up.")

    def train_elastic_width(self, model, ofa_model):
        if self.elastic_width_allowed:
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
        if self.elastic_depth_allowed:
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
        if self.elastic_kernels_allowed == True:
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

    def train_elastic_dilation(self, model, ofa_model):
        if self.elastic_dilation_allowed == True:
            # train elastic kernels
            for current_dilation_step in range(self.dilation_step_count):
                if current_dilation_step == 0:
                    # step 0 is the full model, and was processed during warm-up
                    continue
                # add a kernel step
                ofa_model.progressive_shrinking_add_dilation()
                self.rebuild_trainer(
                    f"kernel_{current_dilation_step}", self.epochs_dilation_step
                )
                self.trainer.fit(model)
            logging.info("OFA completed dilation matrices.")

    def eval_elastic_width(
        self,
        method_stack,
        method_index,
        lightning_model,
        model,
        trainer_path,
        loginfo_output,
        metrics_output,
        metrics_csv,
    ):
        model.reset_all_widths()
        method = method_stack[method_index]

        for current_width_step in range(self.width_step_count):
            if current_width_step > 0:
                # iteration 0 is the full model with no stepping
                model.step_down_all_channels()

            trainer_path_tmp = trainer_path + f"W {current_width_step}, "
            loginfo_output_tmp = loginfo_output + f"Width {current_width_step}, "
            metrics_output_tmp = metrics_output + f"{current_width_step}, "

            metrics_csv = method(
                method_stack,
                method_index + 1,
                lightning_model,
                model,
                trainer_path_tmp,
                loginfo_output_tmp,
                metrics_output_tmp,
                metrics_csv,
            )

        return metrics_csv

    def eval_elastic_kernel(
        self,
        method_stack,
        method_index,
        lightning_model,
        model,
        trainer_path,
        loginfo_output,
        metrics_output,
        metrics_csv,
    ):
        model.reset_all_kernel_sizes()
        method = method_stack[method_index]

        for current_kernel_step in range(self.kernel_step_count):
            if current_kernel_step > 0:
                # iteration 0 is the full model with no stepping
                model.step_down_all_kernels()

            trainer_path_tmp = trainer_path + f"K {current_kernel_step}, "
            loginfo_output_tmp = loginfo_output + f"Kernel {current_kernel_step}, "
            metrics_output_tmp = metrics_output + f"{current_kernel_step}, "

            metrics_csv = method(
                method_stack,
                method_index + 1,
                lightning_model,
                model,
                trainer_path_tmp,
                loginfo_output_tmp,
                metrics_output_tmp,
                metrics_csv,
            )

        return metrics_csv

    def eval_elastic_dilation(
        self,
        method_stack,
        method_index,
        lightning_model,
        model,
        trainer_path,
        loginfo_output,
        metrics_output,
        metrics_csv,
    ):
        model.reset_all_dilation_sizes()
        method = method_stack[method_index]

        for current_dilation_step in range(self.dilation_step_count):
            if current_dilation_step > 0:
                # iteration 0 is the full model with no stepping
                model.step_down_all_dilations()

            trainer_path_tmp = trainer_path + f"K {current_dilation_step}, "
            loginfo_output_tmp = loginfo_output + f"Dilation {current_dilation_step}, "
            metrics_output_tmp = metrics_output + f"{current_dilation_step}, "

            metrics_csv = method(
                method_stack,
                method_index + 1,
                lightning_model,
                model,
                trainer_path_tmp,
                loginfo_output_tmp,
                metrics_output_tmp,
                metrics_csv,
            )

        return metrics_csv

    def eval_elatic_depth(
        self,
        method_stack,
        method_index,
        lightning_model,
        model,
        trainer_path,
        loginfo_output,
        metrics_output,
        metrics_csv,
    ):
        model.reset_active_depth()
        method = method_stack[method_index]

        for current_depth_step in range(self.depth_step_count):
            if current_depth_step > 0:
                # iteration 0 is the full model with no stepping
                model.active_depth -= 1

            trainer_path_tmp = trainer_path + f"D {current_depth_step}, "
            loginfo_output_tmp = loginfo_output + f"Depth {current_depth_step}, "
            metrics_output_tmp = metrics_output + f"{current_depth_step}, "

            metrics_csv = method(
                method_stack,
                method_index + 1,
                lightning_model,
                model,
                trainer_path_tmp,
                loginfo_output_tmp,
                metrics_output_tmp,
                metrics_csv,
            )

        return metrics_csv

    def eval_single_model(
        self,
        method_stack,
        method_index,
        lightning_model,
        model,
        trainer_path,
        loginfo_output,
        metrics_output,
        metrics_csv,
    ):
        self.rebuild_trainer(trainer_path)
        logging.info(loginfo_output)
        model.reset_validaton_model()
        validation_results = self.trainer.validate(
            lightning_model, ckpt_path=None, verbose=True
        )
        model.reset_validaton_model()

        metrics_csv += metrics_output
        results = validation_results[0]
        torch_params = model.get_validation_model_weight_count()
        metrics_csv += f"{results['val_accuracy']}, {results['total_macs']}, {results['total_weights']}, {torch_params}"
        metrics_csv += "\n"
        return metrics_csv

    # cycle through submodels, test them, store results (under a given width step)
    def eval_model(self, lightning_model, model):
        # disable sampling in forward during evaluation.
        model.eval_mode = True
        # reset_seed()  # run_training does this (?)
        # reset target values to step through

        eval_methods = list()

        if self.elastic_width_allowed:
            eval_methods.append(self.eval_elastic_width)

        if self.elastic_kernels_allowed:
            eval_methods.append(self.eval_elastic_kernel)

        if self.elastic_dilation_allowed:
            eval_methods.append(self.eval_elastic_dilation)

        if self.elastic_depth_allowed:
            eval_methods.append(self.eval_elatic_depth)

        if len(eval_methods) > 0:
            eval_methods.append(self.eval_single_model)
            self.submodel_metrics_csv = eval_methods[0](
                eval_methods,
                1,
                lightning_model,
                model,
                "Eval ",
                "OFA validating ",
                "",
                self.submodel_metrics_csv,
            )

        if self.random_evaluate:
            self.eval_random_combination(lightning_model, model)

        model.eval_mode = False

    def eval_random_combination(self, lightning_model, model):
        # sample a few random combinations
        model.reset_validaton_model()
        random_eval_number = self.random_eval_number
        prev_max_kernel = model.sampling_max_kernel_step
        prev_max_depth = model.sampling_max_depth_step
        prev_max_width = model.sampling_max_width_step
        prev_max_dilation = model.sampling_max_dilation_step
        model.sampling_max_kernel_step = model.ofa_steps_kernel - 1
        model.sampling_max_dilation_step = model.ofa_steps_dilation - 1
        model.sampling_max_depth_step = model.ofa_steps_depth - 1
        model.sampling_max_width_step = model.ofa_steps_width - 1
        for i in range(random_eval_number):
            random_state = model.sample_subnetwork()

            loginfo_output = f"OFA validating random sample:\n{random_state}"
            trainer_path = f"Eval random sample: "
            metrics_output = ""

            if self.elastic_width_allowed:
                selected_widths = random_state["width_steps"]
                selected_widths_string = str(selected_widths).replace(",", ";")
                metrics_output += f"{selected_widths_string}, "
                trainer_path += f"Ws {selected_widths}, "

            if self.elastic_kernels_allowed:
                selected_kernels = random_state["kernel_steps"]
                selected_kernels_string = str(selected_kernels).replace(",", ";")
                metrics_output += f" {selected_kernels_string}, "
                trainer_path += f"Ks {selected_kernels}, "

            if self.elastic_dilation_allowed:
                selected_dilations = random_state["dilation_steps"]
                selected_dilations_string = str(selected_dilations).replace(",", ";")
                metrics_output += f" {selected_dilations_string}, "
                trainer_path += f"Dils {selected_dilations}, "

            if self.elastic_depth_allowed:
                selected_depth = random_state["depth_step"]
                trainer_path += f"D {selected_depth}, "
                metrics_output += f"{selected_depth}, "

            self.random_metrics_csv = self.eval_single_model(
                None,
                None,
                lightning_model,
                model,
                trainer_path,
                loginfo_output,
                metrics_output,
                self.random_metrics_csv,
            )

        # revert to normal operation after eval.
        model.sampling_max_kernel_step = prev_max_kernel
        model.sampling_max_dilation_step = prev_max_dilation
        model.sampling_max_depth_step = prev_max_depth
        model.sampling_max_width_step = prev_max_width

    def rebuild_trainer(self, step_name: str, epochs: int = 1):
        logger = TensorBoardLogger(".", version=step_name)
        callbacks = common_callbacks(self.config)
        self.trainer = instantiate(
            self.config.trainer, callbacks=callbacks, logger=logger, max_epochs=epochs
        )
