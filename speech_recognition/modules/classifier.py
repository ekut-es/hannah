import logging
import os
import json
import copy

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.classification.precision_recall import Precision
from pytorch_lightning.metrics import Accuracy, Recall, F1, ROC, ConfusionMatrix
from pytorch_lightning.loggers import TensorBoardLogger, LoggerCollection
from pytorch_lightning.metrics.metric import MetricCollection
from torch._C import Value
from .config_utils import get_loss_function, get_model
from typing import Optional

from speech_recognition.datasets.base import ctc_collate_fn

import tabulate
import torch
import torch.utils.data as data
from torchaudio.transforms import TimeStretch, TimeMasking, FrequencyMasking
from hydra.utils import instantiate, get_class

from ..datasets.NoiseDataset import NoiseDataset
from ..datasets.DatasetSplit import DatasetSplit
from ..datasets.Downsample import Downsample
from ..datasets import AsynchronousLoader, SpeechDataset
from .metrics import Error, plot_confusion_matrix
from ..models.factory.qat import QAT_MODULE_MAPPINGS


from omegaconf import DictConfig


class StreamClassifierModule(LightningModule):
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
    ):
        super().__init__()

        self.save_hyperparameters()
        self.msglogger = logging.getLogger()
        self.initialized = False
        self.train_set = None
        self.test_set = None
        self.dev_set = None
        self.logged_samples = 0

    def prepare_data(self):
        # get all the necessary data stuff
        if not self.train_set or not self.test_set or not self.dev_set:
            get_class(self.hparams.dataset.cls).download(self.hparams.dataset)
            NoiseDataset.download_noise(self.hparams.dataset)
            DatasetSplit.split_data(self.hparams.dataset)
            Downsample.downsample(self.hparams.dataset)

    def setup(self, stage):
        # TODO stage variable is not used!
        self.msglogger.info("Setting up model")
        if self.logger:
            self.logger.log_hyperparams(self.hparams)

        if self.initialized:
            return

        self.initialized = True

        # trainset needed to set values in hparams
        self.train_set, self.dev_set, self.test_set = get_class(
            self.hparams.dataset.cls
        ).splits(self.hparams.dataset)

        # Create example input
        device = self.device
        self.example_input_array = torch.zeros(
            1, self.train_set.channels, self.train_set.input_length
        )
        dummy_input = self.example_input_array.to(device)

        # Instantiate features
        self.features = instantiate(self.hparams.features)
        self.features.to(device)

        features = self._extract_features(dummy_input)
        self.example_feature_array = features

        # Instantiate normalizer
        if self.hparams.normalizer is not None:
            self.normalizer = instantiate(self.hparams.normalizer)
        else:
            self.normalizer = torch.nn.Identity()

        # Instantiate Model
        self.num_classes = len(self.train_set.label_names)
        if hasattr(self.hparams.model, "_target_") and self.hparams.model._target_:
            print(self.hparams.model._target_)
            self.model = instantiate(
                self.hparams.model,
                input_shape=self.example_feature_array.shape,
                labels=self.num_classes,
            )
        else:
            self.hparams.model.width = self.example_feature_array.size(2)
            self.hparams.model.height = self.example_feature_array.size(1)
            self.hparams.model.n_labels = self.num_classes
            self.model = get_model(self.hparams.model)

        # loss function
        self.criterion = get_loss_function(self.model, self.hparams)

        # Metrics
        self.train_metrics = MetricCollection({"train_accuracy": Accuracy()})
        self.val_metrics = MetricCollection(
            {
                "val_accuracy": Accuracy(),
                "val_error": Error(),
                "val_recall": Recall(num_classes=self.num_classes, average="weighted"),
                "val_precision": Precision(
                    num_classes=self.num_classes, average="weighted"
                ),
                "val_f1": F1(num_classes=self.num_classes, average="weighted"),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "test_accuracy": Accuracy(),
                "test_error": Error(),
                "test_recall": Recall(num_classes=self.num_classes, average="weighted"),
                "test_precision": Precision(
                    num_classes=self.num_classes, average="weighted"
                ),
                "rest_f1": F1(num_classes=self.num_classes, average="weighted"),
            }
        )

        self.test_confusion = ConfusionMatrix(num_classes=self.num_classes)
        self.test_roc = ROC(num_classes=self.num_classes, compute_on_step=False)

        augmentation_passes = []
        if self.hparams.time_masking > 0:
            augmentation_passes.append(TimeMasking(self.hparams.time_masking))
        if self.hparams.frequency_masking > 0:
            augmentation_passes.append(TimeMasking(self.hparams.frequency_masking))

        if augmentation_passes:
            self.augmentation = torch.nn.Sequential(*augmentation_passes)
        else:
            self.augmentation = torch.nn.Identity()

    @property
    def total_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
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

    def calculate_batch_metrics(self, output, y, loss, metrics, prefix):
        if isinstance(output, list):
            for idx, out in enumerate(output):
                out = torch.nn.functional.softmax(out, dim=1)
                metrics(out, y)
                self.log_dict(metrics)
        else:
            try:
                output = torch.nn.functional.softmax(output, dim=1)
                metrics(output, y)
                self.log_dict(metrics)
            except ValueError:
                logging.critical("Could not calculate batch metrics: {outputs}")
        self.log(f"{prefix}_loss", loss)

    @staticmethod
    def get_balancing_sampler(dataset):
        distribution = dataset.class_counts
        weights = 1.0 / torch.tensor(
            [distribution[i] for i in range(len(distribution))], dtype=torch.float
        )

        sampler_weights = weights[dataset.get_label_list()]

        sampler = data.WeightedRandomSampler(sampler_weights, len(dataset))
        return sampler

    # TRAINING CODE
    def training_step(self, batch, batch_idx):
        x, x_len, y, y_len = batch

        output = self(x)
        y = y.view(-1)
        loss = self.criterion(output, y)

        # METRICS
        self.calculate_batch_metrics(output, y, loss, self.train_metrics, "train")

        return loss

    def train_dataloader(self):
        train_batch_size = self.hparams["batch_size"]
        dataset_conf = self.hparams.dataset
        sampler = None
        sampler_type = dataset_conf.get("sampler", "random")
        if sampler_type == "weighted":
            sampler = self.get_balancing_sampler(self.train_set)
        else:
            sampler = data.RandomSampler(self.train_set)

        train_loader = data.DataLoader(
            self.train_set,
            batch_size=train_batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=self.hparams["num_workers"],
            collate_fn=ctc_collate_fn,
            sampler=sampler,
            multiprocessing_context="fork" if self.hparams["num_workers"] > 0 else None,
        )

        # if self.device.type == "cuda":
        #    train_loader = AsynchronousLoader(train_loader, device=self.device)

        self.batches_per_epoch = len(train_loader)

        return train_loader

    def on_train_epoch_end(self, outputs):
        self.eval()
        self._log_weight_distribution()
        self.train()

    # VALIDATION CODE
    def validation_step(self, batch, batch_idx):

        # dataloader provides these four entries per batch
        x, x_length, y, y_length = batch

        # INFERENCE
        output = self(x)
        y = y.view(-1)
        loss = self.criterion(output, y)

        # METRICS
        self.calculate_batch_metrics(output, y, loss, self.val_metrics, "val")
        return loss

    def val_dataloader(self):

        dev_loader = data.DataLoader(
            self.dev_set,
            batch_size=min(len(self.dev_set), 16),
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            collate_fn=ctc_collate_fn,
            multiprocessing_context="fork" if self.hparams["num_workers"] > 0 else None,
        )

        # if self.device.type == "cuda":
        #    dev_loader = AsynchronousLoader(dev_loader, device=self.device)

        return dev_loader

    # TEST CODE
    def test_step(self, batch, batch_idx):

        # dataloader provides these four entries per batch
        x, x_length, y, y_length = batch

        output = self(x)
        y = y.view(-1)
        loss = self.criterion(output, y)

        # METRICS
        self.calculate_batch_metrics(output, y, loss, self.test_metrics, "test")
        logits = torch.nn.functional.softmax(output, dim=1)
        self.test_confusion(logits, y)
        self.test_roc(logits, y)

        if isinstance(self.test_set, SpeechDataset):
            self._log_audio(x, logits, y)

        return loss

    def test_dataloader(self):
        test_loader = data.DataLoader(
            self.test_set,
            batch_size=min(len(self.test_set), 16),
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            collate_fn=ctc_collate_fn,
            multiprocessing_context="fork" if self.hparams["num_workers"] > 0 else None,
        )

        # if self.device.type == "cuda":
        #    test_loader = AsynchronousLoader(test_loader, device=self.device)

        return test_loader

    def on_test_end(self) -> None:
        if self.trainer and self.trainer.fast_dev_run:
            return

        metric_table = []
        for name, metric in self.test_metrics.items():
            metric_table.append((name, metric.compute().item()))

        logging.info("\nTest Metrics:\n%s", tabulate.tabulate(metric_table))

        confusion_matrix = self.test_confusion.compute()
        self.test_confusion.reset()

        confusion_plot = plot_confusion_matrix(
            confusion_matrix.cpu().numpy(), self.test_set.class_names
        )
        confusion_plot.savefig("test_confusion.png")
        confusion_plot.savefig("test_confusion.pdf")

        # roc_fpr, roc_tpr, roc_thresholds = self.test_roc.compute()
        self.test_roc.reset()

    def _extract_features(self, x):
        x = self.features(x)

        if x.dim() == 4:
            new_channels = x.size(1) * x.size(2)
            x = torch.reshape(x, (x.size(0), new_channels, x.size(3)))

        return x

    def forward(self, x):
        x = self._extract_features(x)
        if self.training:
            x = self.augmentation(x)
        x = self.normalizer(x)
        x = self.model(x)
        return x

    def save(self):

        logging.info("saving weights to json...")
        output_dir = "."
        filename = os.path.join(output_dir, "model.json")
        state_dict = self.model.state_dict()
        with open(filename, "w") as f:
            json.dump(state_dict, f, default=lambda x: x.tolist(), indent=2)

        quantized_model = copy.deepcopy(self.model)
        quantized_model.cpu()
        if hasattr(self.model, "qconfig") and self.model.qconfig:
            quantized_model = torch.quantization.convert(
                quantized_model, mapping=QAT_MODULE_MAPPINGS, remove_qconfig=True
            )

        logging.info("saving onnx...")
        try:
            dummy_input = self.example_feature_array.cpu()

            torch.onnx.export(
                quantized_model,
                dummy_input,
                os.path.join(output_dir, "model.onnx"),
                verbose=False,
                opset_version=13,
            )
        except Exception as e:
            logging.error("Could not export onnx model ...\n {}".format(str(e)))

        # logging.info("Saving torchscript model ...")
        # try:
        #     dummy_input = self.example_feature_array.cpu()
        #     traced_model = torch.jit.trace(quantized_model, (dummy_input,))
        #     traced_model.save(os.path.join(output_dir, "model.pt"))

        # except Exception as e:
        #     logging.error("Could not export torchscript model ...\n {}".format(str(e)))

    # CALLBACKS
    def on_fit_end(self):
        loggers = self._logger_iterator()
        for logger in loggers:
            if isinstance(logger, TensorBoardLogger):
                items = map(lambda x: (x[0], x[1].compute()), self.val_metrics.items())
                logger.log_hyperparams(self.hparams, metrics=dict(items))

    def _log_weight_distribution(self):
        for name, params in self.named_parameters():
            loggers = self._logger_iterator()

            for logger in loggers:
                if hasattr(logger.experiment, "add_histogram"):
                    try:
                        logger.experiment.add_histogram(
                            name, params, self.current_epoch
                        )
                    except ValueError as e:
                        logging.critical("Could not add histogram for param %s", name)

        for name, module in self.named_modules():
            loggers = self._logger_iterator()
            if hasattr(module, "scaled_weight"):
                for logger in loggers:
                    if hasattr(logger.experiment, "add_histogram"):
                        try:
                            logger.experiment.add_histogram(
                                f"{name}.scaled_weight",
                                module.scaled_weight,
                                self.current_epoch,
                            )
                        except ValueError as e:
                            logging.critical(
                                "Could not add histogram for param %s", name
                            )

    def _log_audio(self, x, logits, y):
        prediction = torch.argmax(logits, dim=1)
        correct = prediction == y
        for num, result in enumerate(correct):
            if not result and self.logged_samples < 10:
                loggers = self._logger_iterator()
                class_names = self.test_set.class_names
                for logger in loggers:
                    if hasattr(logger.experiment, "add_audio"):
                        logger.experiment.add_audio(
                            f"sample{self.logged_samples}_{class_names[prediction[num]]}_{class_names[y[num]]}",
                            x[num],
                            self.current_epoch,
                            self.test_set.samplingrate,
                        )
                self.logged_samples += 1

    def _logger_iterator(self):
        if isinstance(self.logger, LoggerCollection):
            loggers = self.logger
        else:
            loggers = [self.logger]

        return loggers


class SpeechClassifierModule(LightningModule):
    def __init__(self, *args, **kwargs):
        logging.critical(
            "SpeechClassifierModule has been renamed to StreamClassifierModule"
        )
        super(SpeechClassifierModule, self).__init__(*args, **kwargs)
