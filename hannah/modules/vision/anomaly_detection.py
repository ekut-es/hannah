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
import json
import logging
import os
import statistics
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from hydra.utils import get_class, instantiate
from tqdm import trange

from ..augmentation.batch_augmentation import BatchAugmentationPipeline
from .base import VisionBaseModule
from .loss import SemiSupervisedLoss

msglogger = logging.getLogger(__name__)


class AnomalyDetectionModule(VisionBaseModule):
    def compute_anomaly_score(self):
        anomaly_score = None
        largest_train_error = None
        if self.train_losses:
            largest_train_error = torch.max(
                torch.stack(self.train_losses), dim=0
            ).values
            self.normalized_train_errors = torch.stack(self.train_losses) / (
                largest_train_error
            )
            anomaly_score = np.percentile(
                self.normalized_train_errors.cpu().numpy(), 90
            )
            largest_train_error = largest_train_error.detach().cpu().numpy()
        return anomaly_score, largest_train_error

    def common_step(self, step_name, batch, batch_idx):
        ss_loss = SemiSupervisedLoss()
        batch = self._decode_batch(batch)
        x = batch["data"]
        labels = batch.get("labels", None)

        if batch_idx == 0:
            self._log_batch_images("input", batch_idx, x)

        mean = self.train_set.mean
        std = self.train_set.std
        seq = BatchAugmentationPipeline(
            {
                "Normalize": {"mean": mean, "std": std},
            }
        )
        normalized_data = seq.forward(x)
        prediction_result = self.forward(normalized_data)

        loss = torch.tensor([0.0], device=self.device)
        preds = None

        if self.hparams.train_val_loss == "classifier" or step_name == "test":
            loss = ss_loss.forward(logits=prediction_result.logits, labels=labels)
            self.log(f"{step_name}_classifier_loss", loss, batch_size=self.batch_size)
            preds = torch.argmax(prediction_result.logits, dim=1)
            self.metrics[f"{step_name}_metrics"](preds, labels)
            self.log_dict(
                self.metrics[f"{step_name}_metrics"], batch_size=self.batch_size
            )

        elif (
            self.hparams.train_val_loss == "decoder"
            and prediction_result.decoded is not None
        ):
            decoded = prediction_result.decoded
            current_loss = ss_loss.forward(true_data=x, decoded=decoded)
            self.log(
                f"{step_name}_decoder_loss", current_loss, batch_size=self.batch_size
            )
            loss += current_loss

            if batch_idx == 0:
                self._log_batch_images("decoded", batch_idx, decoded)

        self.log(f"{step_name}_loss", loss, batch_size=self.batch_size)
        return loss, prediction_result, batch, preds

    def training_step(self, batch, batch_idx):
        batch_list = [batch]
        if isinstance(batch, dict) and "unlabeled" in batch:
            (
                batch_unlabeled,
                batch_normal_labeled,
                batch_anomaly_labeled,
            ) = self.identify_batches(batch)
            batch_list = [batch_unlabeled, batch_normal_labeled, batch_anomaly_labeled]

        loss = torch.tensor([0.0], device=self.device)

        for batch in batch_list:
            batch = self._decode_batch(batch)
            x = batch["data"]
            labels = batch.get("labels", None)
            boxes = batch.get("bbox", [])

            if batch_idx == 0:
                self._log_batch_images("input", batch_idx, x)

            if (
                torch.is_tensor(labels) and torch.numel(labels) == 0
            ):  # case that no anomalies are in the batch, prevent loss to be nan
                pass

            else:
                current_loss = torch.tensor([0.0], device=self.device)
                augmented_data, x = self.augment(x, labels, boxes, batch_idx)
                prediction_result = self.forward(augmented_data)
                if not self.hparams.train_val_loss:
                    raise ValueError(
                        "Please specify the desired loss function for training with the anomaly detection module"
                    )

                elif (
                    self.hparams.train_val_loss == "decoder"
                    and prediction_result.decoded is not None
                ):
                    decoded = prediction_result.decoded
                    if torch.is_tensor(labels) and torch.sum(labels) > 0:  # anomaly
                        ss_loss = SemiSupervisedLoss(
                            kind="normal"
                        )  # change back to anomaly
                    else:
                        ss_loss = SemiSupervisedLoss(kind="normal")
                    current_loss = ss_loss.forward(true_data=x, decoded=decoded)
                    self.train_losses.append(current_loss)
                    if batch_idx == 0:
                        self._log_batch_images("decoded", batch_idx, decoded)

                elif (
                    self.hparams.train_val_loss == "classifier"
                    and labels is not None
                    and prediction_result.logits.numel() > 0
                ):
                    ss_loss = SemiSupervisedLoss(kind="normal")
                    logits = prediction_result.logits
                    current_loss = ss_loss.forward(
                        logits=logits, labels=labels, weight=self.loss_weights
                    )
                    preds = torch.argmax(logits, dim=1)
                    self.metrics["train_metrics"](preds, labels)
                    self.log_dict(
                        self.metrics["train_metrics"], batch_size=self.batch_size
                    )

                self.log(
                    "train_" + f"{self.hparams.train_val_loss}_loss",
                    current_loss,
                    batch_size=self.batch_size,
                )
                loss += current_loss
                self.log("train_loss", loss, batch_size=self.batch_size)

            loss += current_loss
        return loss

    def validation_step(self, batch, batch_idx):
        loss, prediction_result, batch, preds = self.common_step(
            "val", batch, batch_idx
        )
        labels = batch.get("labels", None)
        if self.hparams.train_val_loss == "decoder":
            anomaly_score, largest_train_error = self.compute_anomaly_score()
            if anomaly_score:
                if (loss.cpu().numpy() / largest_train_error) > anomaly_score:
                    preds = torch.empty(
                        size=labels.size(), device=labels.device, dtype=labels.dtype
                    ).fill_(1)
                else:
                    preds = torch.empty(
                        size=labels.size(), device=labels.device, dtype=labels.dtype
                    ).fill_(0)
                self.metrics["val_metrics"](preds, labels)
                self.log_dict(self.metrics["val_metrics"], batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        loss, step_results, batch, preds = self.common_step("test", batch, batch_idx)
        y = batch.get("labels", None)

        # if ~torch.isnan(loss):
        #    self.test_losses.extend(loss)
        # preds = step_results.preds

        if y is not None and preds is not None:
            self.test_confusion(preds, y)
            self.predictions = torch.cat((self.predictions.to(preds.device), preds), 0)
            self.labels = torch.cat((self.labels.to(y.device), y), 0)

    def on_test_end(self):
        """wd_dir = os.getcwd()
        score, largest_train_error = self.compute_anomaly_score()
        train_errors = self.normalized_train_errors
        plt.hist(train_errors.detach().cpu().numpy(), bins=100)
        plt.axvline(score, linestyle="dashed")
        plt.title("Normalized train reconstruction errors")
        plt.savefig(wd_dir + "/normalized_train_errors.png")

        test = (
            torch.tensor(self.test_losses, device=self.device)
            / torch.max(torch.stack(self.train_losses), dim=0).values
        )
        plt.hist(test.detach().cpu().numpy(), bins=100)
        plt.title("Normalized test reconstruction errors")
        plt.savefig(wd_dir + "/normalized_test_errors.png")
        print("Anomaly score", score)
        print(
            "Largest train error",
            torch.max(torch.stack(self.train_losses), dim=0).values,
        )"""

        self._plot_confusion_matrix()
        self._AUROC(self.predictions, self.labels)

    def on_train_epoch_end(self):
        self.train_losses = self.train_losses[-1000:]
        super().on_train_epoch_end()

    def identify_batches(self, batch):
        batch_labeled = batch["labeled"]
        batch_unlabeled = batch["unlabeled"]

        normal_labeled_idx = (batch_labeled["labels"] == 0).nonzero(as_tuple=True)[0]
        labels_normal = batch_labeled["labels"][normal_labeled_idx]
        batch_normal_labeled = {
            "data": torch.index_select(batch_labeled["data"], 0, normal_labeled_idx),
            "labels": labels_normal,
            "bbox": [],
        }

        anomaly_labeled_idx = (batch_labeled["labels"] == 1).nonzero(as_tuple=True)[0]
        labels_anomaly = batch_labeled["labels"][anomaly_labeled_idx]
        batch_anomaly_labeled = {
            "data": torch.index_select(batch_labeled["data"], 0, anomaly_labeled_idx),
            "labels": labels_anomaly,
            "bbox": [
                batch_labeled.get("bbox")[i] for i in anomaly_labeled_idx.tolist()
            ],
        }
        return batch_unlabeled, batch_normal_labeled, batch_anomaly_labeled
