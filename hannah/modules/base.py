import copy
import io
import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional, Type, TypeVar

import tabulate
import torch
import torch.utils.data as data
import torchvision
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import (
    LightningLoggerBase,
    LoggerCollection,
    TensorBoardLogger,
)
from pytorch_lightning.utilities.distributed import rank_zero_only
from torchmetrics import MetricCollection

from ..models.factory.qat import QAT_MODULE_MAPPINGS
from ..utils import fullname
from .metrics import plot_confusion_matrix

msglogger: logging.Logger = logging.getLogger(__name__)


class ClassifierModule(LightningModule, ABC):
    def __init__(
        self,
        dataset: DictConfig,
        model: DictConfig,
        optimizer: DictConfig,
        features: DictConfig,
        num_workers: int = 0,
        batch_size: int = 128,
        time_masking: int = 0,
        frequency_masking: int = 0,
        scheduler: Optional[DictConfig] = None,
        normalizer: Optional[DictConfig] = None,
        export_onnx: bool = True,
        export_relay: bool = False,
        gpus=None,
        shuffle_all_dataloaders: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.initialized = False
        self.train_set = None
        self.test_set = None
        self.dev_set = None
        self.logged_samples = 0
        self.export_onnx = export_onnx
        self.gpus = gpus
        self.shuffle_all_dataloaders = shuffle_all_dataloaders

        self.val_metrics: MetricCollection = MetricCollection({})
        self.test_metrics: MetricCollection = MetricCollection({})
        self.train_metrics: MetricCollection = MetricCollection({})

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

    def on_train_start(self) -> None:
        super().on_train_start()

        input_array = self.example_input_array.clone().to(self.device)

        for logger in self._logger_iterator():
            if hasattr(logger, "log_graph"):
                logger.log_graph(self, input_array)

    def configure_optimizers(self) -> Dict[str, Any]:
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
                    except ValueError:
                        logging.critical("Could not add histogram for param %s", name)

        for name, module in self.named_modules():
            loggers = self._logger_iterator()

            if hasattr(module, "running_var") and module.running_var is not None:
                for logger in loggers:
                    if hasattr(logger.experiment, "add_histogram"):
                        try:
                            logger.experiment.add_histogram(
                                f"{name}.running_var",
                                module.running_var,
                                self.current_epoch,
                            )
                        except ValueError:
                            logging.critical(
                                "Could not add histogram for param %s", name
                            )

            if hasattr(module, "scale_factor"):
                for logger in loggers:
                    if hasattr(logger.experiment, "add_histogram"):
                        try:
                            logger.experiment.add_histogram(
                                f"{name}.scale_factor",
                                module.scale_factor,
                                self.current_epoch,
                            )
                        except ValueError:
                            logging.critical(
                                "Could not add histogram for param %s", name
                            )

            if hasattr(module, "scaled_weight"):
                for logger in loggers:
                    if hasattr(logger.experiment, "add_histogram"):
                        try:
                            logger.experiment.add_histogram(
                                f"{name}.scaled_weight",
                                module.scaled_weight,
                                self.current_epoch,
                            )
                        except ValueError:
                            logging.critical(
                                "Could not add histogram for param %s", name
                            )

    def _logger_iterator(self) -> Iterable[LightningLoggerBase]:
        loggers = []
        if self.trainer:
            loggers = self.trainer.loggers

        return loggers

    @staticmethod
    def get_balancing_sampler(dataset) -> Any:
        distribution = dataset.class_counts
        weights = 1.0 / torch.tensor(
            [distribution[i] for i in range(len(distribution))], dtype=torch.float
        )

        sampler_weights = weights[dataset.get_label_list()]

        sampler = data.WeightedRandomSampler(sampler_weights, len(dataset))
        return sampler

    @rank_zero_only
    def save(self):
        output_dir = "."
        quantized_model = copy.deepcopy(self.model)
        quantized_model.cpu()
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
