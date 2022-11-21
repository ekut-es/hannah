#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
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
import os
from typing import Sequence

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.utils
from hydra.utils import get_class, instantiate
from sklearn.metrics import auc, roc_curve
from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
)

from ..utils.utils import set_deterministic
from .augmentation.batch_augmentation import BatchAugmentationPipeline
from .base import ClassifierModule
from .metrics import Error

msglogger = logging.getLogger(__name__)


class AnomalyDetectionModule(ClassifierModule):
    def setup(self, stage):
        if self.trainer:
            for logger in self.trainer.loggers:
                logger.log_hyperparams(self.hparams)

        if self.initialized:
            return

        self.initialized = True

        dataset_cls = get_class(self.hparams.dataset.cls)
        self.train_set, self.dev_set, self.test_set = dataset_cls.splits(
            self.hparams.dataset
        )

        if self.hparams.unlabeled_data:
            unlabeled_cls = get_class(self.hparams.unlabeled_data.cls)
            self.train_set_unlabeled, _, _ = unlabeled_cls.splits(
                self.hparams.unlabeled_data
            )

        example_data = self._decode_batch(self.test_set[0])["data"]

        if not isinstance(example_data, torch.Tensor):
            example_data = torch.tensor(example_data, device=self.device)

        self.example_input_array = example_data.clone().detach().unsqueeze(0)
        self.example_feature_array = example_data.clone().detach().unsqueeze(0)

        self.num_classes = 0
        if self.train_set.class_names:
            self.num_classes = len(self.train_set.class_names)

        msglogger.info("Setting up model %s", self.hparams.model.name)
        self.model = instantiate(
            self.hparams.model,
            input_shape=self.example_input_array.shape,
            labels=self.num_classes,
            _recursive_=False,
        )

        if self.hparams.dataset.get("weighted_loss", False) is True:
            loss_weights = torch.tensor(self.train_set.weights)
            loss_weights *= len(self.train_set) / self.num_classes

            msglogger.info("Using weighted loss with weights:")
            for num, (weight, name) in enumerate(
                zip(loss_weights, self.train_set.class_names)
            ):
                msglogger.info("- %s [%d]: %f", name, num, weight.item())

            self.register_buffer("loss_weights", loss_weights)
        else:
            self.loss_weights = None

        self.mixup_fn = self.train_set.get_mixup_fn()
        self.batch_augment_fn = self.train_set.get_batch_augment_fn()

        # setup list for train losses to compute anomaly score
        self.train_losses = list()
        self.normalized_train_errors = None
        self.predictions = list()
        self.labels = list()
        self.test_losses = list()

        # Setup Metrics
        metrics = {}
        if self.num_classes > 0:
            self.test_confusion = ConfusionMatrix(num_classes=self.num_classes)

            for step_name in ["train", "val", "test"]:
                step_metrics = MetricCollection(
                    {
                        f"{step_name}_accuracy": Accuracy(num_classes=self.num_classes),
                        f"{step_name}_error": Error(num_classes=self.num_classes),
                        f"{step_name}_precision_micro": Precision(
                            num_classes=self.num_classes, average="micro"
                        ),
                        f"{step_name}_recall_micro": Recall(
                            num_classes=self.num_classes, average="micro"
                        ),
                        f"{step_name}_f1_micro": F1Score(
                            num_classes=self.num_classes, average="micro"
                        ),
                        f"{step_name}_precision_macro": Precision(
                            num_classes=self.num_classes, average="macro"
                        ),
                        f"{step_name}_recall_macro": Recall(
                            num_classes=self.num_classes, average="macro"
                        ),
                        f"{step_name}_f1_macro": F1Score(
                            num_classes=self.num_classes, average="macro"
                        ),
                    }
                )
                metrics[f"{step_name}_metrics"] = step_metrics

        self.metrics = torch.nn.ModuleDict(metrics)

    def _decode_batch(self, batch):
        if isinstance(batch, Sequence):
            assert len(batch) == 2
            ret = {"data": batch[0], "labels": batch[1]}
        else:
            ret = batch

        return ret

    def get_class_names(self):
        return self.train_set.class_names

    def prepare_data(self):
        # get all the necessary data stuff
        if not self.train_set or not self.test_set or not self.dev_set:
            get_class(self.hparams.dataset.cls).prepare(self.hparams.dataset)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, prediction_result, x, step_name, batch_idx, loss):

        if prediction_result.decoded is not None:
            decoded = prediction_result.decoded
            decoder_loss = F.mse_loss(decoded, x)
            # print(f"{step_name}_decoder_loss", decoder_loss)
            self.log(f"{step_name}_decoder_loss", decoder_loss)
            loss += decoder_loss

            if batch_idx == 0:
                self._log_batch_images("decoded", batch_idx, decoded)

        self.log(f"{step_name}_loss", loss)

        return loss

    def compute_anomaly_score(self):

        anomaly_score = None
        largest_train_error = None
        # lowest_train_error = None
        if self.train_losses:
            largest_train_error = torch.max(
                torch.stack(self.train_losses), dim=0
            ).values
            # lowest_train_error = torch.max(torch.stack(self.train_losses), dim=0).values
            self.normalized_train_errors = torch.stack(self.train_losses) / (
                largest_train_error
            )
            anomaly_score = np.percentile(
                self.normalized_train_errors.cpu().numpy(), 85
            )
            largest_train_error = largest_train_error.detach().cpu().numpy()
        return anomaly_score, largest_train_error

    def common_step(self, step_name, batch, batch_idx):

        batch = self._decode_batch(batch)
        x = batch["data"]
        labels = batch.get("labels", None)
        boxes = batch.get("bbox", None)

        if batch_idx == 0:
            self._log_batch_images("input", batch_idx, x)

        prediction_result = self.forward(x)

        loss = torch.tensor([0.0], device=self.device)
        loss = self.compute_loss(prediction_result, x, step_name, batch_idx, loss)

        anomaly_score, largest_train_error = self.compute_anomaly_score()
        preds = None

        if anomaly_score:
            if (loss.cpu().numpy() / largest_train_error) > anomaly_score:
                preds = torch.empty(
                    size=labels.size(), device=labels.device, dtype=labels.dtype
                ).fill_(1)
            else:
                preds = torch.empty(
                    size=labels.size(), device=labels.device, dtype=labels.dtype
                ).fill_(0)

            self.metrics[f"{step_name}_metrics"](preds, labels)
            self.log_dict(self.metrics[f"{step_name}_metrics"])

        return loss, prediction_result, batch, preds

    def augment(self, labels, batch_idx, x):

        # print('x', x)
        # print(labels)
        # print('x',x)
        # augmented_data = K.RandomGaussianNoise(std=0.05, keepdim=True)(x)
        """augmentation = K.AugmentationSequential(
        #K.RandomSharpness(),
        K.RandomGaussianNoise(std=0.05, keepdim=True)
        )"""
        # augmented_data = augmentation(x)
        """augmentation = K.AugmentationSequential(
            K.RandomErasing(scale=(0.2, 0.21), p=1, keepdim=True),
            K.RandomErasing(scale=(0.2, 0.21), p=1, keepdim=True),
            K.RandomErasing(scale=(0.2, 0.21), p=1, keepdim=True)
            )
        augmented_data = augmentation(x)"""
        augmented_data = x
        # f labels is not None:
        #    if self.mixup_fn is not None:
        #        x, mixup_labels = self.mixup_fn(x, labels)

        # if self.batch_augment_fn is not None:
        #    augmented_data = self.batch_augment_fn(data=x)

        if batch_idx == 0:
            self._log_batch_images("augmented", batch_idx, augmented_data)

        return augmented_data, x

    def training_step(self, batch, batch_idx):
        batch_labeled = batch["labeled"]
        batch_unlabeled = batch["unlabeled"]
        normal_labeled_idx = (batch_labeled["labels"] == 0).nonzero(as_tuple=True)[0]
        labels_normal = batch_labeled["labels"][normal_labeled_idx]
        batch_normal_labeled = {
            "data": torch.index_select(batch_labeled["data"], 0, normal_labeled_idx),
            "labels": labels_normal,
        }

        batch_normal_labeled = self._decode_batch(batch_normal_labeled)
        batch_unlabeled = self._decode_batch(batch_unlabeled)

        loss = torch.tensor([0.0], device=self.device)
        for batch in [batch_unlabeled, batch_normal_labeled]:
            x = batch["data"]
            labels = batch.get("labels", None)
            boxes = batch.get("bbox", None)

            if batch_idx == 0:
                self._log_batch_images("input", batch_idx, x)

            mixup_labels = labels
            augmented_data, x = self.augment(labels, batch_idx, x)

            prediction_result = self.forward(augmented_data)

            current_loss = torch.tensor([0.0], device=self.device)
            current_loss = self.compute_loss(
                prediction_result, x, "train", batch_idx, current_loss
            )

            self.train_losses.append(current_loss)

            loss += current_loss

        return loss

    def validation_step(self, batch, batch_idx):
        self.common_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        loss, step_results, batch, preds = self.common_step("test", batch, batch_idx)

        y = batch.get("labels", None)
        if ~torch.isnan(loss):
            self.test_losses.extend(loss)
        # preds = step_results.preds
        if y is not None and preds is not None:
            with set_deterministic(False):
                self.test_confusion(preds, y)

    def on_test_end(self):
        wd_dir = os.getcwd()
        score, largest_train_error = self.compute_anomaly_score()
        train_errors = self.normalized_train_errors
        plt.hist(train_errors.detach().cpu().numpy(), bins=100)
        plt.axvline(score, linestyle="dashed")
        plt.title("Normalized train reconstruction errors")
        plt.savefig(wd_dir + "/normalized_train_errors.png")
        test = (
            torch.tensor(self.test_losses, device="cuda:0")
            / torch.max(torch.stack(self.train_losses), dim=0).values
        )
        plt.hist(test.detach().cpu().numpy(), bins=100)
        plt.title("Normalized test reconstruction errors")
        plt.savefig(wd_dir + "/normalized_test_errors.png")
        self._plot_confusion_matrix()
        print("Anomaly score", score)
        print(
            "Largest train error",
            torch.max(torch.stack(self.train_losses), dim=0).values,
        )

    def on_train_epoch_end(self):
        self.train_losses = self.train_losses[-1000:]
        self.eval()
        self._log_weight_distribution()
        self.train()
