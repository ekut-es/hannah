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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.utils
from hydra.utils import get_class, instantiate
from pytorch_lightning.trainer.supporters import CombinedLoader
from sklearn.metrics import auc, roc_curve
from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
)

from hannah.datasets.collate import vision_collate_fn
from hannah.utils.utils import set_deterministic

from ..augmentation.batch_augmentation import BatchAugmentationPipeline
from ..base import ClassifierModule
from ..metrics import Error
from .base import VisionBaseModule

# from .anomaly_score import AnomalyScore

msglogger = logging.getLogger(__name__)


class AnomalyDetectionModule(VisionBaseModule):
    def compute_loss(self, prediction_result, x, step_name, anomaly, batch_idx, loss):

        if prediction_result.decoded is not None:
            decoded = prediction_result.decoded
            if anomaly == "anomaly":
                decoder_loss = 0.4 * (1 / (F.mse_loss(decoded, x)))
            else:
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

        batch = self._decode_batch(batch)
        x = batch["data"]
        labels = batch.get("labels", None)
        boxes = batch.get("bbox", None)

        if batch_idx == 0:
            self._log_batch_images("input", batch_idx, x)

        prediction_result = self.forward(x)

        loss = torch.tensor([0.0], device=self.device)
        loss = self.compute_loss(prediction_result, x, step_name, "", batch_idx, loss)

        preds = None
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

            self.metrics[f"{step_name}_metrics"](preds, labels)
            self.log_dict(self.metrics[f"{step_name}_metrics"])

        return loss, prediction_result, batch, preds

    def training_step(self, batch, batch_idx):
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
            "bbox": []
            # torch.index_select(
            #    batch_labeled["bbox"],
            #    0,
            #    anomaly_labeled_idx,
            # ),
        }
        batch_normal_labeled = self._decode_batch(batch_normal_labeled)
        # batch_anomaly_labeled = self._decode_batch(batch_anomaly_labeled)
        batch_unlabeled = self._decode_batch(batch_unlabeled)

        boxes = batch.get("bbox", [])

        loss = torch.tensor([0.0], device=self.device)

        for batch in [batch_unlabeled, batch_normal_labeled]:
            x = batch["data"]
            labels = batch.get("labels", None)
            boxes = batch.get("bbox", [])

            if batch_idx == 0:
                self._log_batch_images("input", batch_idx, x)

            current_loss = torch.tensor([0.0], device=self.device)
            if torch.is_tensor(labels) and torch.sum(labels) > 0:  # anomaly
                augmented_data, x = self.augment(x, labels, boxes, batch_idx)
                prediction_result = self.forward(augmented_data)
                current_loss = self.compute_loss(
                    prediction_result, x, "train", "anomaly", batch_idx, current_loss
                )

            elif (
                torch.is_tensor(labels) and torch.numel(labels) == 0
            ):  # case that no anomalies are in the batch, prevent loss to be nan
                pass

            else:  # normal
                augmented_data, x = self.augment(x, labels, boxes, batch_idx)
                prediction_result = self.forward(augmented_data)
                current_loss = self.compute_loss(
                    prediction_result, x, "train", "", batch_idx, current_loss
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
            torch.tensor(self.test_losses, device=self.device)
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
        super().on_train_epoch_end()

    def _get_dataloader(self, dataset, unlabeled_data=None, shuffle=False):
        batch_size = self.hparams["batch_size"]
        dataset_conf = self.hparams.dataset
        sampler = None
        if shuffle:
            sampler_type = dataset_conf.get("sampler", "random")
            if sampler_type == "weighted":
                sampler = self.get_balancing_sampler(dataset)
            else:
                sampler = data.RandomSampler(dataset)

        loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=self.hparams["num_workers"],
            sampler=sampler,
            collate_fn=vision_collate_fn,
            multiprocessing_context="fork" if self.hparams["num_workers"] > 0 else None,
        )
        self.batches_per_epoch = len(loader)

        if unlabeled_data:
            loader_unlabeled = data.DataLoader(
                unlabeled_data,
                batch_size=batch_size,
                drop_last=True,
                num_workers=self.hparams["num_workers"],
                sampler=data.RandomSampler(unlabeled_data),
                multiprocessing_context="fork"
                if self.hparams["num_workers"] > 0
                else None,
            )
            return CombinedLoader({"labeled": loader, "unlabeled": loader_unlabeled})

        return loader
