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
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.seed import reset_seed, seed_everything

from hannah.models.convnet.models import ConvNet
from hannah.nas.core.parametrized import is_parametrized

from ...callbacks.optimization import HydraOptCallback
from ...callbacks.summaries import MacSummaryCallback
from ...utils import clear_outputs, common_callbacks, fullname
from .optimizer.aging_evolution import AgingEvolution
from ..graph_conversion import model_to_graph

msglogger = logging.getLogger(__name__)


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

        self.predictor = None
        # if predictor is not None:
        #    self.predictor = instantiate(predictor, _recursive_=False)

        self.worklist = []
        self.presample = presample

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
            # estimated_metrics = self.predictor.estimate(model)

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


class OFANasTrainer(NASTrainerBase):
    def __init__(
        self,
        parent_config=None,
        epochs_warmup=10,
        epochs_kernel_step=10,
        epochs_depth_step=10,
        epochs_width_step=10,
        epochs_dilation_step=10,
        epochs_grouping_step=10,
        epochs_dsc_step=10,
        epochs_tuning_step=0,
        elastic_kernels_allowed=False,
        elastic_depth_allowed=False,
        elastic_width_allowed=False,
        elastic_dilation_allowed=False,
        elastic_grouping_allowed=False,
        elastic_dsc_allowed=False,
        evaluate=True,
        random_evaluate=True,
        random_eval_number=100,
        extract_model_config=False,
        warmup_model_path="",
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
        self.epochs_grouping_step = epochs_grouping_step
        self.epochs_dsc_step = epochs_dsc_step
        self.epochs_tuning_step = epochs_tuning_step
        self.elastic_kernels_allowed = elastic_kernels_allowed
        self.elastic_depth_allowed = elastic_depth_allowed
        self.elastic_width_allowed = elastic_width_allowed
        self.elastic_dilation_allowed = elastic_dilation_allowed
        self.elastic_grouping_allowed = elastic_grouping_allowed
        self.elastic_dsc_allowed = elastic_dsc_allowed

        self.evaluate = evaluate
        self.random_evaluate = random_evaluate
        self.random_eval_number = random_eval_number
        self.warmup_model_path = warmup_model_path
        self.extract_model_config = extract_model_config

    def run(self):
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
        self.grouping_step_count = ofa_model.ofa_steps_grouping
        self.dsc_step_count = ofa_model.ofa_steps_dsc
        ofa_model.elastic_kernels_allowed = self.elastic_kernels_allowed
        ofa_model.elastic_depth_allowed = self.elastic_depth_allowed
        ofa_model.elastic_width_allowed = self.elastic_width_allowed
        ofa_model.elastic_dilation_allowed = self.elastic_dilation_allowed
        ofa_model.elastic_grouping_allowed = self.elastic_grouping_allowed
        ofa_model.elastic_dsc_allowed = self.elastic_dsc_allowed
        ofa_model.full_config = self.config["model"]

        logging.info("Kernel Steps: %d", self.kernel_step_count)
        logging.info("Depth Steps: %d", self.depth_step_count)
        logging.info("Width Steps: %d", self.width_step_count)
        logging.info("Grouping Steps: %d", self.grouping_step_count)
        logging.info("DSC Steps: %d", self.dsc_step_count)
        # logging.info("dsc: %d", self.grouping_step_count)

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

        if self.elastic_grouping_allowed:
            self.submodel_metrics_csv += "grouping, "
            self.random_metrics_csv += "group_steps, "

        if self.elastic_dsc_allowed:
            self.submodel_metrics_csv += "dsc, "
            self.random_metrics_csv += "dsc, "

        if (
            self.elastic_width_allowed
            | self.elastic_kernels_allowed
            | self.elastic_dilation_allowed
            | self.elastic_depth_allowed
            | self.elastic_grouping_allowed
            | self.elastic_dsc_allowed
        ):
            self.submodel_metrics_csv += (
                "acc, total_macs, total_weights, torch_params\n"
            )
            self.random_metrics_csv += "acc, total_macs, total_weights, torch_params\n"

        # self.random_metrics_csv = "width_steps, depth, kernel_steps, acc, total_macs, total_weights, torch_params\n"

        logging.info("Once for all Model:\n %s", str(ofa_model))
        # TODO Warmup DSC on or off?
        self.warmup(model, ofa_model)
        ofa_model.reset_shrinking()

        self.train_elastic_kernel(model, ofa_model)
        ofa_model.reset_shrinking()
        self.train_elastic_dilation(model, ofa_model)
        ofa_model.reset_shrinking()
        self.train_elastic_depth(model, ofa_model)
        ofa_model.reset_shrinking()
        self.train_elastic_width(model, ofa_model)
        ofa_model.reset_shrinking()
        self.train_elastic_grouping(model, ofa_model)
        ofa_model.reset_shrinking()
        self.train_elastic_dsc(model, ofa_model)
        ofa_model.reset_shrinking()

        if self.evaluate:
            self.eval_model(model, ofa_model)

            if self.random_evaluate:
                # save random metrics
                msglogger.info("\n%s", self.random_metrics_csv)
                with open("OFA_random_sample_metrics.csv", "w") as f:
                    f.write(self.random_metrics_csv)
            # save self.submodel_metrics_csv
            msglogger.info("\n%s", str(self.submodel_metrics_csv))
            with open("OFA_elastic_metrics.csv", "w") as f:
                f.write(self.submodel_metrics_csv)

    def warmup(self, model, ofa_model):
        """
        > The function rebuilds the trainer with the warmup epochs, fits the model,
        validates the model, and then calls the on_warmup_end() function to
        change some internal variables

        :param model: the model to be trained
        :param ofa_model: the model that we want to train
        """
        # warm-up.
        self.rebuild_trainer("warmup", self.epochs_warmup)
        if self.epochs_warmup > 0 and self.warmup_model_path == "":
            self.trainer.fit(model)
            ckpt_path = "best"
        elif self.warmup_model_path != "":
            ckpt_path = self.warmup_model_path
        self.trainer.validate(ckpt_path=ckpt_path, model=model, verbose=True)
        ofa_model.on_warmup_end()
        ofa_model.reset_validation_model()
        msglogger.info("OFA completed warm-up.")

    def train_elastic_width(self, model, ofa_model):
        """
        > The function trains the model for a number of epochs, then adds a width
        step, then trains the model for a number of epochs, then adds a width step,
        and so on

        :param model: the model to train
        :param ofa_model: the model that will be trained
        """
        if self.elastic_width_allowed:
            # train elastic width
            # first, run channel priority computation
            ofa_model.progressive_shrinking_compute_channel_priorities()
            for current_width_step in range(1, self.width_step_count):
                # add a width step
                ofa_model.progressive_shrinking_add_width()
                if self.epochs_width_step > 0:
                    self.rebuild_trainer(
                        f"width_{current_width_step}", self.epochs_width_step
                    )
                    self.trainer.fit(model)
                    ckpt_path = "best"
                    self.trainer.validate(ckpt_path=ckpt_path, verbose=True)
            msglogger.info("OFA completed width steps.")

    def train_elastic_depth(self, model, ofa_model):
        """
        > The function trains the model for a number of epochs, then progressively
        shrinks the depth of the model, and trains the model for a number of epochs
        again

        :param model: the model to train
        :param ofa_model: the model to be trained
        """
        if self.elastic_depth_allowed:
            # train elastic depth
            for current_depth_step in range(1, self.depth_step_count):
                # add a depth reduction step
                ofa_model.progressive_shrinking_add_depth()
                if self.epochs_depth_step > 0:
                    self.rebuild_trainer(
                        f"depth_{current_depth_step}", self.epochs_depth_step
                    )
                    self.trainer.fit(model)
                    ckpt_path = "best"
                    self.trainer.validate(ckpt_path=ckpt_path, verbose=True)
            msglogger.info("OFA completed depth steps.")

    def train_elastic_kernel(self, model, ofa_model):
        """
        > The function trains the elastic kernels by progressively shrinking the
        model and training the model for a number of epochs and repeats this process
        until the number of kernel steps is reached

        :param model: the model to train
        :param ofa_model: the model that will be trained
        """
        if self.elastic_kernels_allowed:
            # train elastic kernels
            for current_kernel_step in range(1, self.kernel_step_count):
                # add a kernel step
                ofa_model.progressive_shrinking_add_kernel()
                if self.epochs_kernel_step > 0:
                    self.rebuild_trainer(
                        f"kernel_{current_kernel_step}", self.epochs_kernel_step
                    )
                    self.trainer.fit(model)
                    ckpt_path = "best"
                    self.trainer.validate(ckpt_path=ckpt_path, verbose=True)
            msglogger.info("OFA completed kernel matrices.")

    def train_elastic_dilation(self, model, ofa_model):
        """
        > The function trains the model for a number of epochs, then adds a dilation
        step, and trains the model for a number of epochs, and repeats this process
        until the number of dilation steps is reached

        :param model: the model to be trained
        :param ofa_model: the model that will be trained
        """
        if self.elastic_dilation_allowed:
            # train elastic kernels
            for current_dilation_step in range(1, self.dilation_step_count):
                # add a kernel step
                ofa_model.progressive_shrinking_add_dilation()
                if self.epochs_dilation_step > 0:
                    self.rebuild_trainer(
                        f"kernel_{current_dilation_step}", self.epochs_dilation_step
                    )
                    self.trainer.fit(model)
                    ckpt_path = "best"
                    self.trainer.validate(ckpt_path=ckpt_path, verbose=True)
            msglogger.info("OFA completed dilation matrices.")

    def train_elastic_grouping(self, model, ofa_model):
        """
        > The function trains the model for a number of epochs, then adds a group
        step, and trains the model for a number of epochs, and repeats this process
        until the number of group steps is reached

        :param model: the model to be trained
        :param ofa_model: the model that will be trained
        """
        if self.elastic_grouping_allowed:
            # train elastic groups
            for current_grouping_step in range(1, self.grouping_step_count):
                # add a group step
                ofa_model.progressive_shrinking_add_group()
                if self.epochs_grouping_step > 0:
                    self.rebuild_trainer(
                        f"group_{current_grouping_step}", self.epochs_grouping_step
                    )
                    self.trainer.fit(model)
                    ckpt_path = "best"
                    self.trainer.validate(ckpt_path=ckpt_path, verbose=True)
            msglogger.info("OFA completed grouping matrices.")

    def train_elastic_dsc(self, model, ofa_model):
        """
        > The function trains the model for a number of epochs, then adds a dsc
        step (turns Depthwise Separable Convolution on and off), and trains the model for a number of epochs, and repeats this process
        until the number of dsc steps is reached

        :param model: the model to be trained
        :param ofa_model: the model that will be trained
        """
        if self.elastic_dsc_allowed is True:
            # train elastic groups
            for current_dsc_step in range(1, self.dsc_step_count):
                # add a group step
                ofa_model.progressive_shrinking_add_dsc()
                if self.epochs_dsc_step > 0:
                    self.rebuild_trainer(
                        f"dsc_{current_dsc_step}", self.epochs_dsc_step
                    )
                    self.trainer.fit(model)
                    ckpt_path = "best"
                    self.trainer.validate(ckpt_path=ckpt_path, verbose=True)
            msglogger.info("OFA completed dsc matrices.")

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
        """
        > This function steps down the width of the model, and then calls the next
        method in the stack

        :param method_stack: a list of methods that will be called in order
        :param method_index: The index of the current method in the method stack
        :param lightning_model: the lightning model to be trained
        :param model: the model to be trained
        :param trainer_path: The path to the trainer
        :param loginfo_output: This is the string that will be printed to the
        console
        :param metrics_output: a string that will be written to the metrics csv file
        :param metrics_csv: a string that contains the metrics for the current model
        :return: The metrics_csv is being returned.
        """
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
        """
        > This function steps down the kernel size of the model, and then calls the
        next method in the stack

        :param method_stack: The list of methods to be called
        :param method_index: The index of the current method in the method stack
        :param lightning_model: the lightning model to be trained
        :param model: the model to be trained
        :param trainer_path: The path to the trainer
        :param loginfo_output: This is the string that will be printed to the
        console
        :param metrics_output: This is the string that will be printed to the
        console
        :param metrics_csv: a string that contains the metrics for the current model
        :return: The metrics_csv is being returned.
        """
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
        """
        > This function evaluates the model with a different dilation size for each
        layer

        :param method_stack: The list of methods to be called
        :param method_index: The index of the method in the method stack
        :param lightning_model: the lightning model to be trained
        :param model: the model to be evaluated
        :param trainer_path: The path to the trainer
        :param loginfo_output: This is the string that will be printed to the
        console
        :param metrics_output: a string that will be written to the metrics csv file
        :param metrics_csv: a string that contains the csv data for the metrics
        :return: The metrics_csv is being returned.
        """
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

    def eval_elastic_depth(
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
        """
        > This function will run the next method in the stack for each depth step,
        and then return the metrics_csv

        :param method_stack: The list of methods to be called
        :param method_index: The index of the current method in the method stack
        :param lightning_model: the lightning model to be trained
        :param model: The model to be trained
        :param trainer_path: The path to the trainer, which is used to save the
        model
        :param loginfo_output: This is the string that will be printed to the
        console
        :param metrics_output: This is the string that will be printed to the
        console
        :param metrics_csv: This is the CSV file that we're writing to
        :return: The metrics_csv is being returned.
        """
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

    def eval_elastic_grouping(
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
        """
        > This function evaluates the model with a different group size for each
        layer

        :param method_stack: The list of methods to be called
        :param method_index: The index of the method in the method stack
        :param lightning_model: the lightning model to be trained
        :param model: the model to be evaluated
        :param trainer_path: The path to the trainer
        :param loginfo_output: This is the string that will be printed to the
        console
        :param metrics_output: a string that will be written to the metrics csv file
        :param metrics_csv: a string that contains the csv data for the metrics
        :return: The metrics_csv is being returned.
        """
        model.reset_all_group_sizes()
        method = method_stack[method_index]
        for current_group_step in range(self.grouping_step_count):
            if current_group_step > 0:
                # iteration 0 is the full model with no stepping
                model.step_down_all_groups()

            trainer_path_tmp = trainer_path + f"G {current_group_step}, "
            loginfo_output_tmp = loginfo_output + f"Group {current_group_step}, "
            metrics_output_tmp = metrics_output + f"{current_group_step}, "

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

    def eval_elastic_dsc(
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
        """
        > This function evaluates the model with a different dsc  for each
        layer

        :param method_stack: The list of methods to be called
        :param method_index: The index of the method in the method stack
        :param lightning_model: the lightning model to be trained
        :param model: the model to be evaluated
        :param trainer_path: The path to the trainer
        :param loginfo_output: This is the string that will be printed to the
        console
        :param metrics_output: a string that will be written to the metrics csv file
        :param metrics_csv: a string that contains the csv data for the metrics
        :return: The metrics_csv is being returned.
        """
        model.reset_all_dsc()
        method = method_stack[method_index]
        for current_dsc_step in range(self.dsc_step_count):
            if current_dsc_step > 0:
                # iteration 0 is the full model with no stepping
                model.step_down_all_dsc()

            trainer_path_tmp = trainer_path + f"DSC {current_dsc_step}, "
            loginfo_output_tmp = loginfo_output + f"DSC {current_dsc_step}, "
            metrics_output_tmp = metrics_output + f"{current_dsc_step}, "

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
        """
        > This function takes in a model, a trainer, and a bunch of other stuff,
        evaluates the model and tracks the results in der in the given strings and
        returns a string of metrics

        :param method_stack: The list of methods that we're evaluating
        :param method_index: The index of the method in the method stack
        :param lightning_model: the lightning model that we want to evaluate
        :param model: The model to be evaluated
        :param trainer_path: the path to the trainer object
        :param loginfo_output: This is the string that will be printed to the
        console when the model is being evaluated
        :param metrics_output: This is the string that will be written to the
        metrics file. It contains the method name, the method index, and the method
        stack
        :param metrics_csv: a string that will be written to a csv file
        :return: The metrics_csv is being returned.
        """
        self.rebuild_trainer(trainer_path, self.epochs_tuning_step, tensorboard=False)
        msglogger.info(loginfo_output)

        validation_model = model.build_validation_model()

        lightning_model.model = validation_model
        assert model.eval_mode is True

        if self.epochs_tuning_step > 0:
            self.trainer.fit(lightning_model)

        validation_results = self.trainer.validate(
            lightning_model, ckpt_path=None, verbose=True
        )

        lightning_model.model = model

        metrics_csv += metrics_output
        results = validation_results[0]
        torch_params = model.get_validation_model_weight_count()
        metrics_csv += f"{results['val_accuracy']}, {results['total_macs']}, {results['total_weights']}, {torch_params}"
        metrics_csv += "\n"
        return metrics_csv

    def eval_model(self, lightning_model, model):
        """
        First the method stack for the evaluation ist build and then it is according to this evaluated

        :param lightning_model: the lightning model
        :param model: the model to be evaluated
        """
        # disable sampling in forward during evaluation.
        model.eval_mode = True

        eval_methods = []

        if self.elastic_width_allowed:
            eval_methods.append(self.eval_elastic_width)

        if self.elastic_kernels_allowed:
            eval_methods.append(self.eval_elastic_kernel)

        if self.elastic_dilation_allowed:
            eval_methods.append(self.eval_elastic_dilation)

        if self.elastic_depth_allowed:
            eval_methods.append(self.eval_elastic_depth)

        if self.elastic_grouping_allowed:
            eval_methods.append(self.eval_elastic_grouping)

        if self.elastic_dsc_allowed:
            eval_methods.append(self.eval_elastic_dsc)

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

        random_eval_number = self.random_eval_number
        prev_max_kernel = model.sampling_max_kernel_step
        prev_max_depth = model.sampling_max_depth_step
        prev_max_width = model.sampling_max_width_step
        prev_max_dilation = model.sampling_max_dilation_step
        prev_max_grouping = model.sampling_max_grouping_step
        prev_max_dsc = model.sampling_max_dsc_step
        model.sampling_max_kernel_step = model.ofa_steps_kernel - 1
        model.sampling_max_dilation_step = model.ofa_steps_dilation - 1
        model.sampling_max_depth_step = model.ofa_steps_depth - 1
        model.sampling_max_width_step = model.ofa_steps_width - 1
        model.sampling_max_grouping_step = model.ofa_steps_grouping - 1
        model.sampling_max_dsc_step = model.ofa_steps_dsc - 1
        assert model.eval_mode is True
        for i in range(random_eval_number):
            model.reset_validation_model()
            random_state = model.sample_subnetwork()

            loginfo_output = f"OFA validating random sample:\n{random_state}"
            trainer_path = "Eval random sample: "
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

            if self.elastic_grouping_allowed:
                selected_groups = random_state["grouping_steps"]
                selected_groups_string = str(selected_groups).replace(",", ";")
                metrics_output += f" {selected_groups_string}, "
                trainer_path += f"Gs {selected_groups_string}, "

            if self.elastic_dsc_allowed:
                selected_dscs = random_state["dsc_steps"]
                selected_dscs_string = str(selected_dscs).replace(",", ";")
                metrics_output += f" {selected_dscs_string}, "
                trainer_path += f"DSCs {selected_dscs_string}, "

            if self.elastic_depth_allowed:
                selected_depth = random_state["depth_step"]
                trainer_path += f"D {selected_depth}, "
                metrics_output += f"{selected_depth}, "
            if self.extract_model_config:
                model.print_config("r" + str(i))

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
        model.sampling_max_grouping_step = prev_max_grouping
        model.sampling_max_dsc_step = prev_max_dsc

    def rebuild_trainer(
        self, step_name: str, epochs: int = 1, tensorboard: bool = True
    ) -> Trainer:
        if tensorboard:
            logger = TensorBoardLogger(".", version=step_name)
        else:
            logger = CSVLogger(".", version=step_name)
        callbacks = common_callbacks(self.config)
        self.trainer = instantiate(
            self.config.trainer, callbacks=callbacks, logger=logger, max_epochs=epochs
        )
