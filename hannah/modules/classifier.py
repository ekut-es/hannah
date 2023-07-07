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

import logging
import math
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
    AUROC,
    ROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
)

from hannah.datasets.collate import ctc_collate_fn

from ..datasets import SpeechDataset
from ..models.factory.qat import QAT_MODULE_MAPPINGS
from ..utils.utils import set_deterministic
from .base import ClassifierModule
from .config_utils import get_loss_function, get_model
from .metrics import Error

msglogger = logging.getLogger(__name__)


class BaseStreamClassifierModule(ClassifierModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_data(self):
        # get all the necessary data stuff
        if not self.train_set or not self.test_set or not self.dev_set:
            if self.hparams.dataset:
                get_class(self.hparams.dataset.cls).prepare(self.hparams.dataset)

    def setup(self, stage):
        # TODO stage variable is not used!
        msglogger.info("Setting up model")
        if self._trainer:
            for logger in self.trainer.loggers:
                logger.log_hyperparams(self.hparams)

        if self.initialized:
            msglogger.info("Model setup already completed skipping setup")
            return

        self.initialized = True

        if self.hparams.dataset is not None:

            # trainset needed to set values in hparams
            self.train_set, self.dev_set, self.test_set = self.get_split()

            self.num_classes = int(self.get_num_classes())
            self.dataset_type = "binary" if self.num_classes == 2 else "multiclass"

        # Create example input
        device = self.device
        if self.example_input_array is None:
            self.example_input_array = self.get_example_input_array()
        dummy_input = self.example_input_array

        logging.info("Example input array shape: %s", str(dummy_input.shape))

        # Instantiate features
        self.features = instantiate(self.hparams.features)

        features = self._extract_features(dummy_input)
        self.example_feature_array = features

        # Instantiate normalizer
        if self.hparams.normalizer is not None:
            self.normalizer = instantiate(self.hparams.normalizer)
        else:
            self.normalizer = torch.nn.Identity()

        self.example_feature_array = self.normalizer(self.example_feature_array)

        # Instantiate Model
        if hasattr(self.hparams, "model"):
            if hasattr(self.hparams.model, "_target_") and self.hparams.model._target_:
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
        self.train_metrics = MetricCollection(
            {
                "train_accuracy": Accuracy(task=self.dataset_type, num_classes=self.num_classes, weight='macro'),
                "train_error": Accuracy(task=self.dataset_type, num_classes=self.num_classes, weight='macro'),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "val_accuracy": Accuracy(task=self.dataset_type, num_classes=self.num_classes, weight='macro'),
                "val_error": Error(task=self.dataset_type, num_classes=self.num_classes, weight='macro'),
                "val_recall": Recall(task=self.dataset_type, num_classes=self.num_classes, weight='macro'),
                "val_precision": Precision(task=self.dataset_type, num_classes=self.num_classes, weight='macro'),
                "val_f1": F1Score(task=self.dataset_type, num_classes=self.num_classes, weight='macro'),
                "val_auroc": AUROC(task=self.dataset_type, num_classes=self.num_classes, weight='macro'),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "test_accuracy": Accuracy(task=self.dataset_type, num_classes=self.num_classes, weight='macro'),
                "test_error": Error(task=self.dataset_type, num_classes=self.num_classes, weight='macro'),
                "test_recall": Recall(task=self.dataset_type, num_classes=self.num_classes, weight='macro'),
                "test_precision": Precision(task=self.dataset_type, num_classes=self.num_classes, weight='macro'),
                "test_f1": F1Score(task=self.dataset_type, num_classes=self.num_classes, weight='macro'),
                "test_auroc": AUROC(task=self.dataset_type, num_classes=self.num_classes, weight='macro'),
            }
        )

        self.test_confusion = ConfusionMatrix(
            task=self.dataset_type, num_classes=self.num_classes
        )
        self.test_roc = ROC(
            task=self.dataset_type, num_classes=self.num_classes, compute_on_step=False
        )

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
                if self.dataset_type == "binary":
                    out = out.argmax(dim=1)
                metrics(out, y)
                self.log_dict(metrics, batch_size=self.batch_size)
        else:
            try:
                output = torch.nn.functional.softmax(output, dim=1)
                if self.dataset_type == "binary":
                    output = output.argmax(dim=1)
                metrics(output, y)
                self.log_dict(metrics, batch_size=self.batch_size)
            except ValueError as e:
                logging.critical(f"Could not calculate batch metrics: output={output}")

            self.log(f"{prefix}_loss", loss, batch_size=self.batch_size)

    # TRAINING CODE
    def training_step(self, batch, batch_idx):
        if len(batch) == 4:
            x, x_length, y, y_length = batch
        elif len(batch) == 2:
            x, y = batch
        output = self(x)
        y = y.view(-1)
        loss = self.criterion(output, y)
        # METRICS
        self.calculate_batch_metrics(output, y, loss, self.train_metrics, "train")

        return loss

    # @abstractmethod
    # def train_dataloader(self):
    #    pass

    def on_train_epoch_end(self):
        self.eval()
        self._log_weight_distribution()
        self.train()

    # VALIDATION CODE
    def validation_step(self, batch, batch_idx):

        # dataloader provides these four entries per batch
        if len(batch) == 4:
            x, x_length, y, y_length = batch
        elif len(batch) == 2:
            x, y = batch

        # INFERENCE
        output = self(x)
        # print(output)
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
            shuffle=self.shuffle_all_dataloaders,
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
            shuffle=self.shuffle_all_dataloaders,
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
                            x[num].permute(
                                1, 0
                            ),  # Need to permute for tensorboard #FIXME: test with other loggers
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
        if self.train_set is not None:
            return torch.zeros(1, *self.train_set.size())
        else:
            return self.example_input_array

    def train_dataloader(self):
        return self._get_dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.dev_set)

    def test_dataloader(self):
        return self._get_dataloader(self.test_set)


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
            yield self._get_dataloader(train_set, shuffle=True)

    def val_dataloader(self):
        for dev_set in self.dev_set:
            yield self._get_dataloader(dev_set)

    def test_dataloader(self):
        for test_set in self.test_set:
            yield self._get_dataloader(test_set)

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
