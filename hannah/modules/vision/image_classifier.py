#
# Copyright (c) 2024 Hannah contributors.
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
import csv
import logging
import os
from typing import Sequence

import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.utils
from hydra.utils import get_class, instantiate
from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
)

from hannah.datasets.collate import vision_collate_fn

from ..metrics import Error
from .base import VisionBaseModule
from .loss import SemiSupervisedLoss

msglogger = logging.getLogger(__name__)


class ImageClassifierModule(VisionBaseModule):
    def common_step(self, step_name, batch, batch_idx):
        # print("step_name", step_name)
        ss_loss = SemiSupervisedLoss()

        batch = self._decode_batch(batch)

        x = batch["data"]
        labels = batch.get("labels", None)
        boxes = batch.get("bbox", None)

        if batch_idx == 0:
            self._log_batch_images(f"input_{step_name}", batch_idx, x)

        augmented_data, x = self.augment(x, labels, boxes, batch_idx)

        prediction_result = self.forward(augmented_data)

        loss = torch.tensor([0.0], device=self.device)
        preds = None

        if hasattr(prediction_result, "logits"):
            logits = prediction_result.logits
        else:
            logits = prediction_result

        if labels is not None and logits.numel() > 0:
            classifier_loss = ss_loss.forward(
                logits=logits, labels=labels, weight=self.loss_weights
            )

            self.log(
                f"{step_name}_classifier_loss",
                classifier_loss,
                batch_size=self.batch_size,
            )
            loss += classifier_loss
            preds = torch.argmax(logits, dim=1)
            provs = torch.softmax(logits, dim=1)

            # For HMM and Viterbi Post-Processing
            if ((step_name == "train") or (step_name == 'val')) and self.current_epoch == self.trainer.max_epochs-1:
                metadata = batch.get('metadata', {}).copy()
                metadata.update({'preds_cnn': preds.cpu().numpy(), 'labels': labels.cpu().numpy()})
                df = pd.DataFrame(metadata)
                df.to_csv(os.getcwd() + f'_cnn_{step_name}_output', mode='a', index=False, header=True)

            if step_name == 'test':
                metadata = batch.get('metadata', {}).copy()
                metadata.update({'preds_cnn': preds.cpu().numpy(), 'labels': labels.cpu().numpy()})
                df = pd.DataFrame(metadata)
                df.to_csv(os.getcwd() + '_cnn_test_output', mode='a', index=False, header=True)

            self.metrics[f"{step_name}_metrics"](preds, labels)

            self.log_dict(
                self.metrics[f"{step_name}_metrics"], batch_size=self.batch_size
            )

        if (
            hasattr(prediction_result, "decoded")
            and prediction_result.decoded.numel() > 0
        ):
            decoded = prediction_result.decoded
            decoder_loss = torch.nn.functional.mse_loss(
                decoded, augmented_data, reduction="mean"
            )
            self.log(f"{step_name}_decoder_loss", decoder_loss)
            loss += decoder_loss

            if batch_idx == 0:
                self._log_batch_images("decoded", batch_idx, decoded)

        self.log(f"{step_name}_loss", loss, batch_size=self.batch_size, prog_bar=True)
        return loss, prediction_result, batch, preds

    def training_step(self, batch, batch_idx):
        batch_labeled = batch
        batch_unlabeled = None
        if isinstance(batch, dict) and "unlabeled" in batch:
            batch_labeled = batch["labeled"]
            batch_unlabeled = batch["unlabeled"]

        loss, _, _, _ = self.common_step("train", batch_labeled, batch_idx)

        if self.pseudo_label is not None and batch_unlabeled is not None:
            pseudo_loss = self.pseudo_label.training_step(
                batch_unlabeled, self.trainer, self, batch_idx
            )

            self.log("train_pseudo_loss", pseudo_loss, batch_size=self.batch_size)
            self.log("train_supervised_loss", loss, batch_size=self.batch_size)
            loss += pseudo_loss
            self.log("train_combined_loss", loss, batch_size=self.batch_size)
        elif batch_unlabeled is not None:
            msglogger.critical(
                "Batch contains unlabeled data but no pseudo labeling is configured."
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, prediction_result, batch, preds = self.common_step(
            "val", batch, batch_idx
        )

        return None

    def test_step(self, batch, batch_idx):
        _, step_results, batch, preds = self.common_step("test", batch, batch_idx)

        y = batch.get("labels", None)
        if y is not None and preds is not None:
            self.test_confusion(preds, y)

        if (
            y is not None
            and hasattr(step_results, "logits")
            and step_results.logits.numel() > 0
        ):
            probs = torch.softmax(step_results.logits, dim=1)
            self.test_roc(probs, y)
            self.test_pr_curve(probs, y)
