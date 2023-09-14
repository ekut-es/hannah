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
import torch
import torch.utils.data as data
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from sklearn.metrics import auc
from torchaudio.transforms import FrequencyMasking, TimeMasking, TimeStretch
from torchmetrics import Metric, MetricCollection

from hannah.datasets.collate import ctc_collate_fn

from ..models.factory.qat import QAT_MODULE_MAPPINGS
from ..utils import set_deterministic
from .config_utils import get_loss_function, get_model

msglogger = logging.getLogger(__name__)

from .classifier import StreamClassifierModule


class AngleClassifierModule(StreamClassifierModule):
    def setup(self, stage):
        # TODO stage variable is not used!
        msglogger.info("Setting up model")
        if self.logger:
            msglogger.info("Model setup already completed skipping setup")
            self.logger.log_hyperparams(self.hparams)

        if self.initialized:
            return

        self.initialized = True

        if self.hparams.dataset is not None:

            # trainset needed to set values in hparams
            self.train_set, self.dev_set, self.test_set = self.get_split()

            self.num_classes = self.hparams.model.n_labels

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
        self.criterion = self.loss_function

        # Metrics
        self.train_metrics = MetricCollection(
            {
                "train_accuracy": self.get_accuracy_metric(),
                "train_error": self.get_error_metric(),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "val_accuracy": self.get_accuracy_metric(),
                "val_error": self.get_error_metric(),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "test_accuracy": self.get_accuracy_metric(),
                "test_error": self.get_error_metric(),
            }
        )

    def calculate_batch_metrics(self, output, y, loss, metrics, prefix):
        if isinstance(output, list):
            for idx, out in enumerate(output):
                metrics(out, y)
                self.log_dict(metrics, batch_size=self.batch_size)
        else:
            try:
                metrics(output, y)
                self.log_dict(metrics, batch_size=self.batch_size)
            except ValueError:
                logging.critical("Could not calculate batch metrics: {outputs}")
        self.log(f"{prefix}_loss", loss, batch_size=self.batch_size)

    # TRAINING CODE
    def training_step(self, batch, batch_idx):
        x, x_len, y, y_len = batch

        output = self(x)
        loss = self.criterion(output, y)

        # METRICS
        self.calculate_batch_metrics(output, y, loss, self.train_metrics, "train")

        return loss

    # VALIDATION CODE
    def validation_step(self, batch, batch_idx):

        # dataloader provides these four entries per batch
        x, x_length, y, y_length = batch

        # INFERENCE
        output = self(x)
        loss = self.criterion(output, y)

        # METRICS
        self.calculate_batch_metrics(output, y, loss, self.val_metrics, "val")
        return loss

    # TEST CODE
    def test_step(self, batch, batch_idx):

        # dataloader provides these four entries per batch
        x, x_length, y, y_length = batch

        output = self(x)
        loss = self.criterion(output, y)

        # METRICS
        self.calculate_batch_metrics(output, y, loss, self.test_metrics, "test")

        return loss

    def forward(self, x):
        x = self._extract_features(x)
        x = self.model(x)
        return x


class CartesianClassifierModule(AngleClassifierModule):
    @staticmethod
    def get_angle_diff(scores, labels):
        assert scores.shape[0] == labels.shape[0]
        assert scores.shape[1] == 2
        assert labels.shape[1] == 2

        labels_norm = torch.nn.functional.normalize(labels)

        scores_norm = torch.nn.functional.normalize(scores)

        x_hat = labels_norm[:, 0]
        y_hat = labels_norm[:, 1]

        x = scores_norm[:, 0]
        y = scores_norm[:, 1]

        result = torch.acos(x_hat * x + y_hat * y)

        return result

    def get_dist(self, scores, labels):
        assert scores.shape[0] == labels.shape[0]
        assert scores.shape[1] == 2
        assert labels.shape[1] == 2

        x_hat, y_hat = labels[:, 0], labels[:, 1]
        x, y = scores[:, 0], scores[:, 1]

        dist = torch.sqrt(torch.square(x - x_hat) + torch.square(y - y_hat))
        return dist

    def loss_function(self, scores, labels):
        return torch.mean(self.get_dist(scores, labels))

    def get_error_metric(self):
        return self.AngleError()

    def get_accuracy_metric(self):
        return self.AngleAccuracy()

    class AngleError(Metric):
        def __init__(self, dist_sync_on_step=False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)

            self.add_state("errors", default=torch.Tensor())

        def update(self, preds: torch.Tensor, target: torch.Tensor):
            error = CartesianClassifierModule.get_angle_diff(preds, target)
            self.errors = torch.cat((self.errors, error))

        def compute(self):
            return torch.mean(self.errors) / (2 * math.pi) * 360.0

    class AngleAccuracy(AngleError):
        def compute(self):
            return 1.0 - super().compute() / 180.0


class DirectAngleClassifierModule(AngleClassifierModule):
    def forward(self, x):
        x = self._extract_features(x)
        x = self.model(x)
        x = torch.nn.functional.hardtanh(x, min_val=-math.pi, max_val=math.pi)
        return x

    @staticmethod
    def get_angle_diff(scores, labels, e=1e-7):
        assert scores.shape[0] == labels.shape[0]
        assert scores.shape[1] == 1
        assert labels.shape[1] == 2

        labels_norm = torch.nn.functional.normalize(labels)

        x_hat = labels_norm[:, 0]
        y_hat = labels_norm[:, 1]

        x = torch.sin(scores.squeeze())
        y = torch.cos(scores.squeeze())

        result = torch.acos(torch.clamp(x_hat * x + y_hat * y, -1.0 + e, 1.0 - e))

        return result

    def loss_function(self, scores, labels):
        return torch.mean(self.get_angle_diff(scores, labels))

    def get_error_metric(self):
        return self.AngleError()

    def get_accuracy_metric(self):
        return self.AngleAccuracy()

    class AngleError(Metric):
        def __init__(self, dist_sync_on_step=False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)

            self.add_state("errors", default=torch.Tensor())

        def update(self, preds: torch.Tensor, target: torch.Tensor):
            error = DirectAngleClassifierModule.get_angle_diff(preds, target)
            self.errors = torch.cat((self.errors, error))

        def compute(self):
            return torch.mean(self.errors) / (2 * math.pi) * 360.0

    class AngleAccuracy(AngleError):
        def compute(self):
            return 1.0 - super().compute() / 180.0


class SINCOSClassifierModule(AngleClassifierModule):
    def forward(self, x):
        x = self._extract_features(x)
        x = self.model(x)
        x = torch.nn.functional.hardtanh(x, min_val=-1, max_val=1)
        return x

    @staticmethod
    def get_loss(scores, labels):
        assert scores.shape[0] == labels.shape[0]
        assert scores.shape[1] == 2
        assert labels.shape[1] == 2

        labels_norm = torch.nn.functional.normalize(labels)

        sin_hat = labels_norm[:, 0]
        cos_hat = labels_norm[:, 1]

        sin = scores[:, 0]
        cos = scores[:, 1]

        return torch.mean(torch.abs(sin_hat - sin) + torch.abs(cos_hat - cos))

    @staticmethod
    def get_angle_diff(scores, labels, e=1e-7):
        assert scores.shape[0] == labels.shape[0]
        assert scores.shape[1] == 2
        assert labels.shape[1] == 2

        labels_norm = torch.nn.functional.normalize(labels)

        scores_norm = torch.nn.functional.normalize(scores)

        x_hat = labels_norm[:, 0]
        y_hat = labels_norm[:, 1]

        x = scores_norm[:, 0]
        y = scores_norm[:, 1]

        result = torch.acos(torch.clamp(x_hat * x + y_hat * y, -1.0 + e, 1.0 - e))

        return result

    def loss_function(self, scores, labels):
        return torch.mean(SINCOSClassifierModule.get_loss(scores, labels))

    def get_error_metric(self):
        return self.AngleError()

    def get_accuracy_metric(self):
        return self.AngleAccuracy()

    class AngleError(Metric):
        def __init__(self, dist_sync_on_step=False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)

            self.add_state("errors", default=torch.Tensor())

        def update(self, preds: torch.Tensor, target: torch.Tensor):
            error = SINCOSClassifierModule.get_angle_diff(preds, target)
            self.errors = torch.cat((self.errors, error))

        def compute(self):
            return torch.mean(self.errors) / (2 * math.pi) * 360.0

    class AngleAccuracy(AngleError):
        def compute(self):
            return 1.0 - super().compute() / 180.0
