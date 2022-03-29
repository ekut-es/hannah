import logging
import platform
from abc import abstractmethod
from typing import Dict, Optional, Union

import numpy as np
import tabulate
import torch
import torch.utils.data as data
import torchvision
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchaudio.transforms import FrequencyMasking, TimeMasking, TimeStretch
from torchmetrics import (
    F1Score,
    ROC,
    Accuracy,
    ConfusionMatrix,
    MetricCollection,
    Precision,
    Recall,
)

from hannah.datasets.base import ctc_collate_fn

from ..datasets import SpeechDataset
from ..models.factory.qat import QAT_MODULE_MAPPINGS
from ..utils import set_deterministic
from .base import ClassifierModule
from .config_utils import get_loss_function, get_model
from .metrics import Error

logger = logging.getLogger(__name__)


class BaseStreamClassifierModule(ClassifierModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_data(self):
        # get all the necessary data stuff
        if not self.train_set or not self.test_set or not self.dev_set:
            get_class(self.hparams.dataset.cls).prepare(self.hparams.dataset)

    def setup(self, stage):
        # TODO stage variable is not used!
        logger.info("Setting up model")
        if self.logger:
            self.msglogger.info("Model setup already completed skipping setup")
            self.logger.log_hyperparams(self.hparams)

        if self.initialized:
            return

        self.initialized = True

        if self.hparams.dataset is not None:

            # trainset needed to set values in hparams
            self.train_set, self.dev_set, self.test_set = self.get_split()

            self.num_classes = self.get_num_classes()

        # Create example input
        device = self.device
        self.example_input_array = self.get_example_input_array()
        dummy_input = self.example_input_array.to(device)
        logging.info("Example input array shape: %s", str(dummy_input.shape))
        if platform.machine() == "ppc64le":
            dummy_input = dummy_input.to("cuda:" + str(self.gpus[0]))

        # Instantiate features
        self.features = instantiate(self.hparams.features)
        self.features.to(device)
        if platform.machine() == "ppc64le":
            self.features.to("cuda:" + str(self.gpus[0]))

        features = self._extract_features(dummy_input)
        self.example_feature_array = features.to(self.device)

        # Instantiate normalizer
        if self.hparams.normalizer is not None:
            self.normalizer = instantiate(self.hparams.normalizer)
        else:
            self.normalizer = torch.nn.Identity()

        self.example_feature_array = self.normalizer(self.example_feature_array)

        # Instantiate Model
        if hasattr(self.hparams.model, "_target_") and self.hparams.model._target_:
            print(self.hparams.model._target_)
            self.model = instantiate(
                self.hparams.model,
                input_shape=self.example_feature_array.shape,
                labels=self.num_classes,
                _recursive_=False,
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
                "val_recall": Recall(num_classes=self.num_classes),
                "val_precision": Precision(num_classes=self.num_classes),
                "val_f1": F1Score(num_classes=self.num_classes),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "test_accuracy": Accuracy(),
                "test_error": Error(),
                "test_recall": Recall(num_classes=self.num_classes),
                "test_precision": Precision(num_classes=self.num_classes),
                "test_f1": F1Score(num_classes=self.num_classes),
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

    @abstractmethod
    def get_example_input_array(self):
        pass

    @abstractmethod
    def get_split(self):
        pass

    @abstractmethod
    def get_num_classes(self):
        pass

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

    # TRAINING CODE
    def training_step(self, batch, batch_idx):
        x, x_len, y, y_len = batch

        output = self(x)
        y = y.view(-1)
        loss = self.criterion(output, y)

        # METRICS
        self.calculate_batch_metrics(output, y, loss, self.train_metrics, "train")

        return loss

    @abstractmethod
    def train_dataloader(self):
        pass

    def get_train_dataloader_by_set(self, train_set):
        train_batch_size = self.hparams["batch_size"]
        dataset_conf = self.hparams.dataset
        sampler_type = dataset_conf.get("sampler", "random")
        if sampler_type == "weighted":
            sampler = self.get_balancing_sampler(train_set)
        else:
            sampler = data.RandomSampler(train_set)

        train_loader = data.DataLoader(
            train_set,
            batch_size=train_batch_size,
            drop_last=True,
            num_workers=self.hparams["num_workers"],
            collate_fn=ctc_collate_fn,
            sampler=sampler,
            multiprocessing_context="fork" if self.hparams["num_workers"] > 0 else None,
        )

        self.batches_per_epoch = len(train_loader)

        return train_loader

    def on_train_epoch_end(self):
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

    @abstractmethod
    def val_dataloader(self):
        pass

    def get_val_dataloader_by_set(self, dev_set):
        dev_loader = data.DataLoader(
            dev_set,
            batch_size=min(len(dev_set), self.hparams["batch_size"]),
            shuffle=True,
            num_workers=self.hparams["num_workers"],
            collate_fn=ctc_collate_fn,
            multiprocessing_context="fork" if self.hparams["num_workers"] > 0 else None,
        )

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
        with set_deterministic(False):
            self.test_confusion(logits, y)

        self.test_roc(logits, y)

        if isinstance(self.test_set, SpeechDataset):
            self._log_audio(x, logits, y)

        return loss

    @abstractmethod
    def test_dataloader(self):
        pass

    def get_test_dataloader_by_set(self, test_set):
        test_loader = data.DataLoader(
            test_set,
            batch_size=min(len(test_set), self.hparams["batch_size"]),
            shuffle=True,
            num_workers=self.hparams["num_workers"],
            collate_fn=ctc_collate_fn,
            multiprocessing_context="fork" if self.hparams["num_workers"] > 0 else None,
        )

        return test_loader

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


class StreamClassifierModule(BaseStreamClassifierModule):
    def get_class_names(self):
        return self.test_set.class_names

    def get_split(self):
        return get_class(self.hparams.dataset.cls).splits(self.hparams.dataset)

    def get_num_classes(self):
        return len(self.train_set.class_names)

    def get_example_input_array(self):
        return torch.zeros(1, *self.train_set.size())

    def train_dataloader(self):
        return self.get_train_dataloader_by_set(self.train_set)

    def val_dataloader(self):
        return self.get_val_dataloader_by_set(self.dev_set)

    def test_dataloader(self):
        return self.get_test_dataloader_by_set(self.test_set)


class CrossValidationStreamClassifierModule(BaseStreamClassifierModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer_fold_callback = None
        self.sets_by_criteria = None
        self.k_fold = self.hparams.dataset.k_fold
        self.test_end_callback_function = None

    def get_class_names(self):
        return get_class(self.hparams.dataset.cls).get_class_names()

    def get_num_classes(self):
        return get_class(self.hparams.dataset.cls).get_num_classes()

    def get_split(self):
        self.sets_by_criteria = get_class(self.hparams.dataset.cls).splits_cv(
            self.hparams.dataset
        )
        return self.prepare_dataloaders(self.sets_by_criteria)

    def get_example_input_array(self):
        return torch.zeros(
            1, self.sets_by_criteria[0].channels, self.sets_by_criteria[0].input_length
        )

    def prepare_dataloaders(self, sets_by_criteria):
        assert self.k_fold >= len(["train", "val", "test"])

        rng = np.random.default_rng()
        subsets = np.arange(len(sets_by_criteria))
        rng.shuffle(subsets)
        splits = np.array_split(subsets, self.k_fold)

        train_sets, dev_sets, test_sets = [], [], []

        for i in range(self.k_fold):
            test_split = splits[0]
            dev_split = splits[1]
            train_split = np.concatenate(splits[2:]).ravel()

            train_sets += [
                torch.utils.data.ConcatDataset(
                    [sets_by_criteria[i] for i in train_split]
                )
            ]
            dev_sets += [
                torch.utils.data.ConcatDataset([sets_by_criteria[i] for i in dev_split])
            ]
            test_sets += [
                torch.utils.data.ConcatDataset(
                    [sets_by_criteria[i] for i in test_split]
                )
            ]

            splits = splits[1:] + [splits[0]]

        return train_sets, dev_sets, test_sets

    def train_dataloader(self):
        for train_set in self.train_set:
            yield self.get_train_dataloader_by_set(train_set)

    def val_dataloader(self):
        for dev_set in self.dev_set:
            yield self.get_val_dataloader_by_set(dev_set)

    def test_dataloader(self):
        for test_set in self.test_set:
            yield self.get_test_dataloader_by_set(test_set)

    def register_test_end_callback_function(self, function):
        self.test_end_callback_function = function

    def test_end_callback(self, test_metrics):
        self.test_end_callback_function(self, test_metrics)

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        items = super().get_progress_bar_dict()
        items["fold_nr"] = self.trainer_fold_callback()
        return items

    def register_trainer_fold_callback(self, callback):
        self.trainer_fold_callback = callback


class SpeechClassifierModule(StreamClassifierModule):
    def __init__(self, *args, **kwargs):
        logging.critical(
            "SpeechClassifierModule has been renamed to StreamClassifierModule speech classifier module will be removed soon"
        )
        super(SpeechClassifierModule, self).__init__(*args, **kwargs)
