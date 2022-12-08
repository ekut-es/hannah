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

import kornia
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

from hannah.utils.utils import set_deterministic

from ..augmentation.batch_augmentation import BatchAugmentationPipeline
from ..augmentation.transforms.kornia_transforms import A
from ..base import ClassifierModule
from ..metrics import Error

msglogger: logging.Logger = logging.getLogger(__name__)


class VisionBaseModule(ClassifierModule):
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

        # setup lists for reconstruction errors to compute anomaly threshold
        self.train_losses = list()
        self.normalized_train_errors = None
        self.predictions = torch.tensor([], device=self.device)
        self.labels = torch.tensor([], device=self.device)
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
            ret = {"data": batch[0], "labels": batch[1], "bbox": []}
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

    def on_train_epoch_end(self):
        self.eval()
        self._log_weight_distribution()
        self.train()

    def augment(self, images, labels, boxes, batch_idx):
        augmented_data = images
        if (
            torch.numel(images) > 0
        ):  # to circumvent error when tensor is empty (depends on batch size)
            seq = A.PatchSequential(
                A.AugmentationSequential(A.RandomErasing(p=0.75, scale=(0.7, 0.7))),
                patchwise_apply=False,
                grid_size=(8, 8),
            )
            augmented_data = seq(augmented_data)

        # seq = BatchAugmentationPipeline({'RandomGaussianNoise': {'p': 0.4, 'keepdim': True}})
        # augmented_data = seq.forward(augmented_data)

        if batch_idx == 0:
            self._log_batch_images("augmented", batch_idx, augmented_data)

        return augmented_data, images
