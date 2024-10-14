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
import io
import json
import logging
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

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
from pytorch_lightning.utilities import CombinedLoader, rank_zero_only
from torchmetrics import AUROC, MetricCollection

from hannah.nas.export.onnx import to_onnx

from ..models.factory.qat import QAT_MODULE_MAPPINGS
from ..utils.utils import fullname
from .metrics import plot_confusion_matrix

try:
    from hannah_tvm.export import export_relay
except ModuleNotFoundError:
    export_relay = None

msglogger: logging.Logger = logging.getLogger(__name__)


class ClassifierModule(LightningModule, ABC):
    def __init__(
        self,
        dataset: Optional[DictConfig] = None,
        model: Union[DictConfig, nn.Module, None] = None,
        optimizer: Optional[DictConfig] = None,
        features: Optional[DictConfig] = None,
        num_workers: Optional[int] = 0,
        batch_size: int = 128,
        time_masking: int = 0,
        frequency_masking: int = 0,
        scheduler: Optional[DictConfig] = None,
        normalizer: Optional[DictConfig] = None,
        unlabeled_data: Optional[DictConfig] = None,
        export_onnx: bool = True,
        export_relay: bool = False,
        shuffle_all_dataloaders: bool = False,
        augmentation: Optional[DictConfig] = None,
        pseudo_labeling: Optional[DictConfig] = None,
        log_images: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model = None
        ignore = []
        if isinstance(model, nn.Module):
            self.model = model
            ignore.append("model")

        self.save_hyperparameters(ignore=ignore)
        self.initialized = False
        self.train_set = None
        self.test_set = None
        self.dev_set = None

        self.train_set_unlabeled = None
        self.test_set_unlabeled = None
        self.dev_set_unlabeled = None

        self.logged_samples = 0
        self._export_onnx = export_onnx
        self.export_relay = export_relay
        self.shuffle_all_dataloaders = shuffle_all_dataloaders

        self.train_set = None
        self.test_set = None
        self.val_set = None

        self.val_metrics: MetricCollection = MetricCollection({})
        self.test_metrics: MetricCollection = MetricCollection({})
        self.train_metrics: MetricCollection = MetricCollection({})

        self.test_roc = None
        self.test_pr_curve = None

        self.pseudo_label = None
        self.batch_size = batch_size

        self.loss_weights = None

        self._log_images = log_images

    @abstractmethod
    def prepare_data(self) -> Any:
        # get all the necessary data stuff
        pass

    @abstractmethod
    def setup(self, stage) -> Any:
        pass

    @abstractmethod
    def get_class_names(self) -> Sequence[str]:
        pass

    def train_dataloader(self):
        return self._get_dataloader(
            self.train_set,
            self.train_set_unlabeled,
            shuffle=True,
        )

    def test_dataloader(self):
        return self._get_dataloader(self.test_set, self.test_set_unlabeled)

    def val_dataloader(self):
        return self._get_dataloader(self.dev_set, self.dev_set_unlabeled)

    def _get_dataloader(self, dataset, unlabeled_data=None, shuffle=False):
        batch_size = self.hparams["batch_size"]

        def calc_workers(dataset):
            result = (
                num_workers
                if num_workers <= dataset.max_workers or dataset.max_workers == -1
                else dataset.max_workers
            )
            return result

        if hasattr(dataset, "loader"):
            loader = dataset.loader(batch_size, shuffle=shuffle)
        else:
            # FIXME: don't use hparams here
            dataset_conf = self.hparams.dataset
            sampler = None
            if shuffle:
                sampler_type = dataset_conf.get("sampler", "random")
                if sampler_type == "weighted":
                    sampler = self.get_balancing_sampler(dataset)
                else:
                    sampler = data.RandomSampler(dataset)

            num_workers = self.hparams["num_workers"]

            num_workers = calc_workers(dataset)

            loader = data.DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True,
                num_workers=num_workers,
                sampler=sampler if not dataset.sequential else None,
                collate_fn=self.collate_fn,
                multiprocessing_context="fork" if num_workers > 0 else None,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None,
                # pin_memory=True,
            )

        self.batches_per_epoch = len(loader)

        if unlabeled_data:
            if hasattr(unlabeled_data, "loader"):
                unlabeled_data = unlabeled_data.loader(batch_size, shuffle=shuffle)
            else:
                unlabeled_workers = calc_workers(unlabeled_data)
                loader_unlabeled = data.DataLoader(
                    unlabeled_data,
                    batch_size=batch_size,
                    drop_last=True,
                    num_workers=unlabeled_workers,
                    sampler=data.RandomSampler(unlabeled_data)
                    if not unlabeled_data.sequential
                    else None,
                    multiprocessing_context="fork" if unlabeled_workers > 0 else None,
                )

            return CombinedLoader(
                {"labeled": loader, "unlabeled": loader_unlabeled},
                mode="max_size_cycle",
            )

        return loader

    def on_train_start(self) -> None:
        super().on_train_start()

    def configure_optimizers(self) -> Any:
        if self.hparams.optimizer is not None:
            optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        else:
            msglogger.warning("No optimizer found in hyperparameters")
            optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

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
                            name, params, global_step=self.current_epoch
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
        target_list = dataset.label_list
        sampler_weights = [weights[i] for i in target_list]
        sampler = data.WeightedRandomSampler(sampler_weights, len(dataset))
        return sampler

    @rank_zero_only
    def save(self):
        output_dir = "."
        quantized_model = copy.deepcopy(self.model)
        quantized_model.cpu()
        quantized_model.train(False)

        if self.export_relay:
            self._do_export_relay(quantized_model)

        if hasattr(self.model, "qconfig") and self.model.qconfig:
            quantized_model = torch.quantization.convert(
                quantized_model, mapping=QAT_MODULE_MAPPINGS, remove_qconfig=True
            )

        if self._export_onnx:
            self._do_export_onnx(output_dir, quantized_model)

    def _do_export_relay(self, quantized_model):
        if export_relay:
            logging.info("Exporting relay model ...")
            export_relay(quantized_model, self.example_feature_array.cpu())
        else:
            raise Exception(
                "Could not export relay due to missing hannah_tvm please install with `poetry install -E tvm`"
            )

    def _do_export_onnx(self, output: Union[io.BytesIO, str, Path], quantized_model):
        logging.info("saving onnx...")

        if not isinstance(output, io.BytesIO):
            output = Path(output)

            if output.suffix != ".onnx":
                output = output / "model.onnx"

        try:
            dummy_input = self.example_feature_array.cpu()
            torch.onnx.export(
                quantized_model,
                dummy_input,
                output,
                verbose=False,
                opset_version=11,
            )
        except Exception as e:
            logging.error("Could not export onnx model ...\n {}".format(str(e)))

    def export_onnx(self, output: Union[io.BytesIO, str, Path]):
        quantized_model = copy.deepcopy(self.model)
        quantized_model.cpu()
        quantized_model.train(False)

        if hasattr(self.model, "qconfig") and self.model.qconfig:
            quantized_model = torch.quantization.convert(
                quantized_model, mapping=QAT_MODULE_MAPPINGS, remove_qconfig=True
            )

        self._do_export_onnx(output, quantized_model)

    def on_load_checkpoint(self, checkpoint) -> None:
        # Calling self setup to make sure that the model is instantiated
        self.setup("train")

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
        self._plot_curves()

    # FIXME: this is terrible please use standard metrics for auroc scores
    def _AUROC(self, preds, target):
        auroc = AUROC(task="binary")
        auroc_score = auroc(preds, target)
        for logger in self._logger_iterator():
            if isinstance(logger, TensorBoardLogger) and hasattr(self, "test_metrics"):
                logger.log_metrics({"AUROC": auroc_score})

    def _plot_curves(self) -> None:
        os.makedirs("plots", exist_ok=True)
        if self.test_roc is not None:
            roc = self.test_roc.compute()
            fpr, tpr, thresholds = roc

            with open("plots/test_roc.json", "w") as fp:
                json.dump(
                    {
                        "fpr": fpr.cpu().numpy().tolist(),
                        "tpr": tpr.cpu().numpy().tolist(),
                        "thresholds": thresholds.cpu().numpy().tolist(),
                        "categories": list(self.test_set.class_names_abbreviated),
                    },
                    fp,
                )

            fig, ax = self.test_roc.plot(roc)
            fig.savefig("plots/test_roc.png")
            fig.savefig("plots/test_roc.pdf")

            self.test_roc.reset()

        if self.test_pr_curve:
            pr_curve = self.test_pr_curve.compute()
            precision, recall, thresholds = pr_curve
            with open("plots/test_pr_curve.json", "w") as fp:
                json.dump(
                    {
                        "precision": precision.cpu().numpy().tolist(),
                        "recall": recall.cpu().numpy().tolist(),
                        "thresholds": thresholds.cpu().numpy().tolist(),
                        "categories": list(self.test_set.class_names_abbreviated),
                    },
                    fp,
                )

            fig, ax = self.test_pr_curve.plot(pr_curve)
            fig.savefig("plots/test_pr_curve.png")
            fig.savefig("plots/test_pr_curve.pdf")

            self.test_pr_curve.reset()

    def _plot_confusion_matrix(self) -> None:
        os.makedirs("plots", exist_ok=True)
        if hasattr(self, "test_confusion") and self.test_confusion is not None:
            confusion_matrix = self.test_confusion.compute()
            self.test_confusion.reset()

            if self.trainer.global_rank > 0:
                return

            json.dump(
                {
                    "data": confusion_matrix.cpu().numpy().tolist(),
                    "categories": list(self.test_set.class_names_abbreviated),
                },
                open("plots/test_confusion.json", "w"),
            )

            confusion_plot = plot_confusion_matrix(
                confusion_matrix.cpu().numpy(),
                categories=self.test_set.class_names_abbreviated,
                figsize=(self.num_classes, self.num_classes),
            )

            confusion_plot.savefig("plots/test_confusion.png")
            confusion_plot.savefig("plots/test_confusion.pdf")

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
        if not self._log_images:
            return
        loggers = self._logger_iterator()
        for logger in loggers:
            if hasattr(logger.experiment, "add_image"):
                if torch.numel(data) > 0:
                    images = torchvision.utils.make_grid(data, normalize=True)
                    logger.experiment.add_image(f"{name}_{batch_idx}", images)

    def _setup_loss_weights(self):
        """Calculate loss weights depending on class frequencies in training set"""
        if not self.hparams.dataset:
            return None

        if self.hparams.dataset.get("weighted_loss", False) is True:
            loss_weights = torch.tensor(self.train_set.weights)
            loss_weights *= len(self.train_set) / self.num_classes

            msglogger.info("Using weighted loss with weights:")
            for num, (weight, name) in enumerate(
                zip(loss_weights, self.train_set.class_names)
            ):
                msglogger.info("- %s [%d]: %f", name, num, weight.item())

            return loss_weights

        return None

    def collate_fn(self, batch):
        return torch.utils.data.default_collate(batch)
