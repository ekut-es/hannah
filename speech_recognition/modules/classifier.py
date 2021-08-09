import logging
import os
import json
import copy
import platform

from abc import abstractmethod

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

from ..datasets import AsynchronousLoader, SpeechDataset
from .metrics import Error, plot_confusion_matrix
from ..models.factory.qat import QAT_MODULE_MAPPINGS

from omegaconf import DictConfig

from numpy import mean


class ClassifierModule(LightningModule):
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

    @abstractmethod
    def prepare_data(self):
        # get all the necessary data stuff
        pass

    @abstractmethod
    def setup(self, stage):
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

    def save(self):
        output_dir = "."
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

    def on_load_checkpoint(self, checkpoint):
        for k, v in self.state_dict().items():
            if k not in checkpoint["state_dict"]:
                self.msglogger.warning(
                    "%s not in state dict using pre initialized values", k
                )
                checkpoint["state_dict"][k] = v


class StreamClassifierModule(ClassifierModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_data(self):
        # get all the necessary data stuff
        if not self.train_set or not self.test_set or not self.dev_set:
            get_class(self.hparams.dataset.cls).prepare(self.hparams.dataset)

    def setup(self, stage):
        # TODO stage variable is not used!
        self.msglogger.info("Setting up model")
        if self.logger:
            self.logger.log_hyperparams(self.hparams)

        if self.initialized:
            return

        self.initialized = True

        if self.hparams.dataset is not None:

            # trainset needed to set values in hparams
            self.train_set, self.dev_set, self.test_set = get_class(
                self.hparams.dataset.cls
            ).splits(self.hparams.dataset)

            self.num_classes = len(self.train_set.class_names)

        # Create example input
        device = self.device
        self.example_input_array = torch.zeros(1, *self.train_set.size())
        dummy_input = self.example_input_array.to(device)
        logging.info("Example input array shape: %s", str(dummy_input.shape))
        if platform.machine() == "ppc64le":
            dummy_input = dummy_input.cuda()

        # Instantiate features
        self.features = instantiate(self.hparams.features)
        self.features.to(device)
        if platform.machine() == "ppc64le":
            self.features.cuda()

        features = self._extract_features(dummy_input)
        self.example_feature_array = features.to(self.device)

        # Instantiate normalizer
        if self.hparams.normalizer is not None:
            self.normalizer = instantiate(self.hparams.normalizer)
        else:
            self.normalizer = torch.nn.Identity()

        # Instantiate Model
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
        # log last loss for given depth if subsampling is present and enabled
        if callable(getattr(self.model, "should_subsample", None)):
            if self.model.should_subsample():
                self.log(f"eloss_{self.model.active_depth}", loss, True)

    # TRAINING CODE
    def training_step(self, batch, batch_idx):
        x, x_len, y, y_len = batch

        # sample active subnet if the relevant function is present
        sample_subnet_function = getattr(self.model, "sample_active_subnet", None)
        if callable(sample_subnet_function):
            self.model.sample_active_subnet()
            self.log("a_depth", self.model.active_depth, True)
            self.log("k_step", self.model.current_kernel_step, True)
            self.log("c_step", self.model.current_channel_step, True)

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

        # if the model has elastic parameters, set them to their defaults for the validation step
        if callable(getattr(self.model, "reset_active_elastic_values", None)):
            self.model.reset_active_elastic_values()

        # INFERENCE
        output = self(x)
        y = y.view(-1)
        loss = self.criterion(output, y)

        if callable(getattr(self.model, "should_subsample", None)):
            if self.model.should_subsample(verify_step=True):
                submodel_output = self.model.get_elastic_depth_output(self.model.active_depth, quantized=False)
                if submodel_output is not None:
                    submodel_loss = self.criterion(submodel_output, y)
                    self.log("eld", submodel_loss-loss, True)
                    # if submodel_loss-loss != 0:
                    #     print(f"Loss value not preserved through extraction: {loss} -> {submodel_loss} (depth={self.model.active_depth})")

        # METRICS
        self.calculate_batch_metrics(output, y, loss, self.val_metrics, "val")

        if callable(getattr(self.model, "resume_active_elastic_values", None)):
            self.model.resume_active_elastic_values()

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

        # if the model has elastic parameters, set them to their defaults for the test step
        if callable(getattr(self.model, "reset_active_elastic_values", None)):
            self.model.reset_active_elastic_values()

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

        # if subsampling depth is present and enabled, compute loss values of elastic depths
        try:
            if callable(getattr(self.model, "should_subsample", None)):
                if self.model.should_subsample():

                    # initialise storage for elastic depth loss values, if not present
                    if getattr(self, "elastic_depth_test_loss_values", None) is None:
                        self.elastic_depth_test_loss_values = []
                        self.test_metrics_elastic_depth = []
                        for i in range(self.model.min_depth, self.model.max_depth+1):
                            self.elastic_depth_test_loss_values.append([])
                            # create a metric for each possible elastic depth submodel
                            self.test_metrics_elastic_depth.append(
                                MetricCollection({
                                    "test_accuracy": Accuracy(),
                                    "test_error": Error(),
                                    "test_recall": Recall(num_classes=self.num_classes, average="weighted"),
                                    "test_precision": Precision(
                                        num_classes=self.num_classes, average="weighted"
                                    ),
                                    "rest_f1": F1(num_classes=self.num_classes, average="weighted"),
                                })
                            )

                    self.model.reset_active_elastic_values()
                    self.model.reset_all_kernel_sizes()
                    for i in range(self.model.min_depth, self.model.max_depth+1):
                        try:
                            submodel_output = self.model.get_elastic_depth_output(i)
                            if submodel_output is None:
                                continue
                            submodel_loss = self.criterion(submodel_output, y)
                            # store the loss value and add output to respective metrics for every depth level
                            self.elastic_depth_test_loss_values[i-self.model.min_depth].append(submodel_loss)
                            self.calculate_batch_metrics(submodel_output, y, submodel_loss, self.test_metrics_elastic_depth[i-self.model.min_depth], "test")
                        except Exception:
                            pass
        except Exception:
            logging.warning("Elastic depth accuracy testing failed.")


        # if subsampling kernels is available and enabled, also compute loss values of elastic kernels
        try:
            if callable(getattr(self.model, "check_kernel_stepping", None)):
                # if self.model.check_kernel_stepping():
                # if kernel stepping is available, try to compute elastic kernel accuracies
                if True:
                    # initialise storage for elastic kernel loss values, if not present
                    if getattr(self, "elastic_kernel_test_loss_values", None) is None:
                        self.elastic_kernel_test_loss_values = []
                        self.test_metrics_elastic_kernel = []

                        # reset elastic values before beginning the counting
                        self.model.reset_active_elastic_values()
                        self.model.reset_all_kernel_sizes()
                        max_kernel_steps = 1
                        # step_down_all_kernels will return false once there were no more kernels left to step through
                        # this will create one metric slot for each possible kernel step, including the initial, full kernels
                        while self.model.step_down_all_kernels() and max_kernel_steps < 255:
                            max_kernel_steps += 1

                        for i in range(max_kernel_steps):
                            self.elastic_kernel_test_loss_values.append([])
                            # create a metric for each possible elastic kernel submodel
                            self.test_metrics_elastic_kernel.append(
                                MetricCollection({
                                    "test_accuracy": Accuracy(),
                                    "test_error": Error(),
                                    "test_recall": Recall(num_classes=self.num_classes, average="weighted"),
                                    "test_precision": Precision(
                                        num_classes=self.num_classes, average="weighted"
                                    ),
                                    "rest_f1": F1(num_classes=self.num_classes, average="weighted"),
                                })
                            )

                    self.model.reset_active_elastic_values()
                    self.model.reset_all_kernel_sizes()
                    # start with full kernels, repeat calculation for each step down
                    for i in range(len(self.test_metrics_elastic_kernel)):
                        try:
                            # with no depth specified, this will pick max depth by default.
                            submodel_output = self.model.get_elastic_depth_output()
                            if submodel_output is None:
                                continue
                            submodel_loss = self.criterion(submodel_output, y)
                            # store the loss value and add output to respective metrics for every depth level
                            self.elastic_kernel_test_loss_values[i-self.model.min_depth].append(submodel_loss)
                            self.calculate_batch_metrics(submodel_output, y, submodel_loss, self.test_metrics_elastic_kernel[i], "test")

                            # step down one kernel size
                            self.model.step_down_all_kernels()
                        except Exception:
                            pass
        except Exception:
            logging.warning("Elastic kernel testing failed.")

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

        try:
            for i in range(len(self.test_metrics_elastic_depth)):
                metric_table = []
                for name, metric in self.test_metrics_elastic_depth[i].items():
                    metric_table.append((name, metric.compute().item()))
                logging.info(f"\nTest Metrics for depth {i+self.model.min_depth}:\n%s", tabulate.tabulate(metric_table))
        except Exception:
            logging.warning("could not compute elastic depth test metrics")

        try:
            for i in range(len(self.test_metrics_elastic_kernel)):
                metric_table = []
                for name, metric in self.test_metrics_elastic_kernel[i].items():
                    metric_table.append((name, metric.compute().item()))
                logging.info(f"\nTest Metrics for kernel step {i}:\n%s", tabulate.tabulate(metric_table))
        except Exception:
            logging.warning("could not compute elastic kernel test metrics")

        try:
            if getattr(self, "elastic_depth_test_loss_values", None) is not None:
                logging.info("Average loss values for elastic depth:")
                for i, vals in enumerate(self.elastic_depth_test_loss_values):
                    average_loss = mean(vals)
                    logging.info(f"{i+self.model.min_depth} : {average_loss}")
        except Exception:
            logging.warning("could not compute elastic depth test loss")

        try:
            if getattr(self, "elastic_kernel_test_loss_values", None) is not None:
                logging.info("Average loss values for elastic kernel:")
                for i, vals in enumerate(self.elastic_kernel_test_loss_values):
                    average_loss = mean(vals)
                    logging.info(f"{i} : {average_loss}")
        except Exception:
            logging.warning("could not compute elastic kernel test loss")

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

        if x.dim() == 4 and self.example_input_array.dim() == 3:
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


class SpeechClassifierModule(StreamClassifierModule):
    def __init__(self, *args, **kwargs):
        logging.critical(
            "SpeechClassifierModule has been renamed to StreamClassifierModule"
        )
        super(SpeechClassifierModule, self).__init__(*args, **kwargs)
