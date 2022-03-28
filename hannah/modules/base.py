import copy
import io
import logging
from multiprocessing import dummy
import os


from abc import abstractmethod, ABC
from typing import Optional
from pytorch_lightning.utilities.distributed import rank_zero_only

import torch
import torch.utils.data as data
import torchvision

from pytorch_lightning.loggers import TensorBoardLogger, LoggerCollection
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import LoggerCollection
from hydra.utils import instantiate

from PIL import Image


from .metrics import plot_confusion_matrix
from ..models.factory.qat import QAT_MODULE_MAPPINGS
from ..utils import fullname

logger = logging.getLogger(__name__)


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
        gpus=None,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.msglogger = logging.getLogger()
        self.initialized = False
        self.train_set = None
        self.test_set = None
        self.dev_set = None
        self.logged_samples = 0
        self.export_onnx = export_onnx
        self.gpus = gpus
        print(dataset.data_folder)

    @abstractmethod
    def prepare_data(self):
        # get all the necessary data stuff
        pass

    @abstractmethod
    def setup(self, stage):
        pass

    @abstractmethod
    def get_class_names(self):
        pass

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())

        retval = {}
        retval["optimizer"] = optimizer

        if self.hparams.scheduler is not None:
            if self.hparams.scheduler._target_ == "torch.optim.lr_scheduler.OneCycleLR":
                scheduler = instantiate(
                    self.hparams.scheduler,
                    optimizer=optimizer,
                    total_steps=self.total_training_steps,
                )
                retval["lr_scheduler"] = dict(scheduler=scheduler, interval="step")
            else:
                scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)

                retval["lr_scheduler"] = dict(scheduler=scheduler, interval="epoch")

        return retval

    @property
    def total_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps > 0:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return int((batches // effective_accum) * self.trainer.max_epochs)

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

    def _logger_iterator(self):
        if isinstance(self.logger, LoggerCollection):
            loggers = self.logger
        else:
            loggers = [self.logger]

        return loggers

    @staticmethod
    def get_balancing_sampler(dataset):
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

    def on_load_checkpoint(self, checkpoint):
        for k, v in self.state_dict().items():
            if k not in checkpoint["state_dict"]:
                self.msglogger.warning(
                    "%s not in state dict using pre initialized values", k
                )
                checkpoint["state_dict"][k] = v

    def on_save_checkpoint(self, checkpoint):
        checkpoint["hyper_parameters"]["_target_"] = fullname(self)

    def on_validation_epoch_end(self):
        for logger in self._logger_iterator():
            if isinstance(logger, TensorBoardLogger) and hasattr(self, "val_metrics"):
                logger.log_hyperparams(
                    self.hparams,
                    {"val_accuracy": self.val_metrics["val_accuracy"].compute().item()},
                )

    def on_test_end(self):

        if self.trainer and self.trainer.fast_dev_run:
            return

        self._plot_confusion_matrix()
        self._plot_roc()

    def _plot_roc(self):
        if hasattr(self, "test_roc"):
            # roc_fpr, roc_tpr, roc_thresholds = self.test_roc.compute()
            self.test_roc.reset()

        if self.trainer.global_rank > 0:
            return

    def _plot_confusion_matrix(self):
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
