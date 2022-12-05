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
from typing import Sequence

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.utils
from hydra.utils import get_class, instantiate

try:
    from timm.data.mixup import Mixup
except ModuleNotFoundError:
    logging.critical("Could not import Mixup from timm.data.mixup")
    Mixup = None
from pytorch_lightning.trainer.supporters import CombinedLoader
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
from ..metrics import Error
from .base import VisionBaseModule

msglogger = logging.getLogger(__name__)


class ImageClassifierModule(VisionBaseModule):
    def common_step(self, step_name, batch, batch_idx):
        # print("step_name", step_name)
        batch = self._decode_batch(batch)

        x = batch["data"]
        labels = batch.get("labels", None)
        boxes = batch.get("bbox", None)

        if batch_idx == 0:
            self._log_batch_images("input", batch_idx, x)

        augmented_data, x = self.augment(x, labels, boxes, batch_idx)

        prediction_result = self.forward(augmented_data)

        loss = torch.tensor([0.0], device=self.device)
        preds = None
        if labels is not None and prediction_result.logits.numel() > 0:
            logits = prediction_result.logits

            classifier_loss = F.cross_entropy(logits, labels, weight=self.loss_weights)

            self.log(f"{step_name}_classifier_loss", classifier_loss)
            loss += classifier_loss

            preds = torch.argmax(logits, dim=1)
            self.metrics[f"{step_name}_metrics"](preds, labels)

            self.log_dict(self.metrics[f"{step_name}_metrics"])

        if prediction_result.decoded.numel() > 0:
            decoded = prediction_result.decoded
            decoder_loss = F.mse_loss(decoded, x)
            self.log(f"{step_name}_decoder_loss", decoder_loss)
            loss += decoder_loss

            if batch_idx == 0:
                self._log_batch_images("decoded", batch_idx, decoded)

        self.log(f"{step_name}_loss", loss)
        return loss, prediction_result, batch, preds

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.common_step("train", batch, batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        self.common_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        _, step_results, batch, preds = self.common_step("test", batch, batch_idx)

        y = batch.get("labels", None)
        if y is not None and preds is not None:
            with set_deterministic(False):
                self.test_confusion(preds, y)

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
