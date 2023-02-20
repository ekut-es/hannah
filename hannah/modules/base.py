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
import io
import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional, Type, TypeVar, Union

import tabulate
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import Logger, TensorBoardLogger
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torchmetrics import AUROC, MetricCollection

from ..models.factory.qat import QAT_MODULE_MAPPINGS
from ..utils.utils import fullname
from .metrics import plot_confusion_matrix

try:
    from hannah_tvm.backend import export_relay
except ModuleNotFoundError:
    export_relay = None


msglogger: logging.Logger = logging.getLogger(__name__)


class ClassifierModule(LightningModule, ABC):
    def __init__(
        self,
        dataset: DictConfig,
        model: Union[DictConfig, nn.Module],
        optimizer: DictConfig,
        features: DictConfig,
        num_workers: int = 0,
        batch_size: int = 128,
        time_masking: int = 0,
        frequency_masking: int = 0,
        scheduler: Optional[DictConfig] = None,
        normalizer: Optional[DictConfig] = None,
        unlabeled_data: Optional[DictConfig] = None,
        export_onnx: bool = True,
        export_relay: bool = False,
        gpus=None,
        shuffle_all_dataloaders: bool = False,
        augmentation: Optional[DictConfig] = None,
        pseudo_labeling: Optional[DictConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        ignore = None
        if not isinstance(model, DictConfig):
            self.model = model
            ignore = ["model"]

        self.save_hyperparameters(ignore=ignore)
        self.initialized = False
        self.train_set = None
        self.test_set = None
        self.dev_set = None

        self.train_set_unlabeled = None
        self.test_set_unlabeled = None
        self.dev_set_unlabeled = None

        self.logged_samples = 0
        self.export_onnx = export_onnx
        self.export_relay = export_relay
        self.gpus = gpus
        self.shuffle_all_dataloaders = shuffle_all_dataloaders

        self.train_set = None
        self.test_set = None
        self.val_set = None

        self.val_metrics: MetricCollection = MetricCollection({})
        self.test_metrics: MetricCollection = MetricCollection({})
        self.train_metrics: MetricCollection = MetricCollection({})

        self.pseudo_label = None
        self.batch_size = batch_size

    @abstractmethod
    def prepare_data(self) -> Any:
        # get all the necessary data stuff
        pass

    @abstractmethod
    def setup(self, stage) -> Any:
        pass

    @abstractmethod
    def get_class_names(self) -> Any:
        pass

    def train_dataloader(self):
        return self._get_dataloader(
            self.train_set, self.train_set_unlabeled, shuffle=True
        )

    def test_dataloader(self):
        return self._get_dataloader(self.test_set, self.test_set_unlabeled)

    def val_dataloader(self):
        return self._get_dataloader(self.dev_set, self.dev_set_unlabeled)

    def _get_dataloader(self, dataset, unlabeled_data=None, shuffle=False):
        dataset_conf = self.hparams.dataset
        sampler = None
        if shuffle:
            sampler_type = dataset_conf.get("sampler", "random")
            if sampler_type == "weighted":
                sampler = self.get_balancing_sampler(dataset)
            else:
                sampler = data.RandomSampler(dataset)

        loader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.hparams["num_workers"],
            sampler=sampler,
            multiprocessing_context="fork" if self.hparams["num_workers"] > 0 else None,
        )
        self.batches_per_epoch = len(loader)

        if unlabeled_data:
            loader_unlabeled = data.DataLoader(
                unlabeled_data,
                batch_size=self.batch_size,
                drop_last=True,
                num_workers=self.hparams["num_workers"],
                sampler=data.RandomSampler(unlabeled_data),
                multiprocessing_context="fork"
                if self.hparams["num_workers"] > 0
                else None,
            )
            return CombinedLoader({"labeled": loader, "unlabeled": loader_unlabeled})

        return loader

    def on_train_start(self) -> None:
        super().on_train_start()

        if hasattr(self, "example_input_array"):
            input_array = self.example_input_array.clone().to(self.device)

            for logger in self._logger_iterator():
                if hasattr(logger, "log_graph"):
                    logger.log_graph(self, input_array)
                    pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())

        retval = {}
        retval["optimizer"] = optimizer

        if self.hparams.scheduler is not None:
            if self.hparams.scheduler._target_ == "torch.optim.lr_scheduler.OneCycleLR":
                scheduler = instantiate(
                    self.hparams.scheduler,
                    optimizer=optimizer,
                    total_steps=self.total_training_steps(),
                )
                retval["lr_scheduler"] = dict(scheduler=scheduler, interval="step")
            else:
                scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)

                retval["lr_scheduler"] = dict(scheduler=scheduler, interval="epoch")

        return retval

    def total_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        estimated_batches = self.trainer.estimated_stepping_batches

        msglogger.debug("Estimated number of training steps: %d", estimated_batches)

        return estimated_batches

    @rank_zero_only
    def _log_weight_distribution(self):
        for name, params in self.named_parameters():
            loggers = self._logger_iterator()

            for logger in loggers:
                if hasattr(logger.experiment, "add_histogram"):
                    try:
                        logger.experiment.add_histogram(
                            name, params, self.current_epoch
                        )
                    except (ValueError, NotImplementedError):
                        logging.critical("Could not add histogram for param %s", name)

    def _logger_iterator(self) -> Iterable[Logger]:
        loggers = []
        if self.trainer:
            loggers = self.trainer.loggers

        return loggers

    def get_balancing_sampler(self, dataset):
        num_sampels = list(dataset.class_counts.values())
        weights = [0 if i is None else 1 / i for i in num_sampels]
        target_list = dataset.get_label_list
        sampler_weights = [weights[i] for i in target_list]
        sampler = data.WeightedRandomSampler(sampler_weights, len(dataset))
        return sampler

    @rank_zero_only
    def save(self):
        output_dir = "."
        quantized_model = copy.deepcopy(self.model)
        quantized_model.cpu()

        if self.export_relay and export_relay:
            logging.info("Exporting relay model ...")
            export_relay(quantized_model, self.example_feature_array.cpu())
        elif self.export_relay:
            raise Exception(
                "Could not export relay due to missing hannah_tvm please install with `poetry install -E tvm`"
            )

        if hasattr(self.model, "qconfig") and self.model.qconfig:
            quantized_model = torch.quantization.convert(
                quantized_model, mapping=QAT_MODULE_MAPPINGS, remove_qconfig=True
            )

        if self.export_onnx:
            logging.info("saving onnx...")
            try:
                dummy_input = self.example_feature_array.cpu()

                torch.onnx.export(
                    quantized_model,
                    dummy_input,
                    os.path.join(output_dir, "model.onnx"),
                    verbose=False,
                    opset_version=11,
                )
            except Exception as e:
                logging.error("Could not export onnx model ...\n {}".format(str(e)))

    def on_load_checkpoint(self, checkpoint) -> None:
        for k, v in self.state_dict().items():
            if k not in checkpoint["state_dict"]:
                msglogger.warning(
                    "%s not in state dict using pre initialized values", k
                )
                checkpoint["state_dict"][k] = v

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["hyper_parameters"]["_target_"] = fullname(self)

    def on_validation_epoch_end(self) -> None:
        if self.trainer:
            if self.trainer.fast_dev_run:
                return
            if self.trainer.sanity_checking:
                return
            if self.trainer.global_rank > 0:
                return
        val_metrics = {}
        for name, metric in self.val_metrics.items():
            val_metrics[name] = metric.compute().item()

        if not val_metrics:
            return

        tabulated_metrics = tabulate.tabulate(
            val_metrics.items(), headers=["Metric", "Value"], tablefmt="github"
        )
        msglogger.info("\nValidation Metrics:\n%s", tabulated_metrics)

        for logger in self._logger_iterator():
            if isinstance(logger, TensorBoardLogger) and hasattr(self, "val_metrics"):
                logger.log_hyperparams(self.hparams, val_metrics)

    def on_test_end(self) -> None:

        if self.trainer and self.trainer.fast_dev_run:
            return

        self._plot_confusion_matrix()
        self._plot_roc()

    def _AUROC(self, preds, target):
        auroc = AUROC(task="binary")
        auroc_score = auroc(preds, target)
        for logger in self._logger_iterator():
            if isinstance(logger, TensorBoardLogger) and hasattr(self, "test_metrics"):
                logger.log_metrics({"AUROC": auroc_score})

    def _plot_roc(self) -> None:
        if hasattr(self, "test_roc"):
            # roc_fpr, roc_tpr, roc_thresholds = self.test_roc.compute()
            self.test_roc.reset()

        if self.trainer.global_rank > 0:
            return

    def _plot_confusion_matrix(self) -> None:
        if hasattr(self, "test_confusion"):
            confusion_matrix = self.test_confusion.compute()
            self.test_confusion.reset()

            if self.trainer.global_rank > 0:
                return

            confusion_plot = plot_confusion_matrix(
                confusion_matrix.cpu().numpy(),
                categories=self.test_set.class_names_abbreviated,
                figsize=(self.num_classes, self.num_classes),
            )

            confusion_plot.savefig("test_confusion.png")
            confusion_plot.savefig("test_confusion.pdf")

            buf = io.BytesIO()

            confusion_plot.savefig(buf, format="jpeg")

            buf.seek(0)
            im = Image.open(buf)
            im = torchvision.transforms.ToTensor()(im)

            loggers = self._logger_iterator()
            for logger in loggers:
                if hasattr(logger.experiment, "add_image"):
                    logger.experiment.add_image(
                        "test_confusion_matrix",
                        im,
                        global_step=self.current_epoch,
                    )

    def _log_batch_images(self, name: str, batch_idx: int, data: torch.tensor):
        loggers = self._logger_iterator()
        for logger in loggers:
            if hasattr(logger.experiment, "add_image"):
                if torch.numel(data) > 0:
                    images = torchvision.utils.make_grid(data, normalize=True)
                    logger.experiment.add_image(f"{name}_{batch_idx}", images)
