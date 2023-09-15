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
from collections import defaultdict
from typing import Optional, Sequence

import kornia
import kornia.augmentation as K
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
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

from ..augmentation.batch_augmentation import BatchAugmentationPipeline
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

        if hasattr(self.hparams, "model"):
            msglogger.info("Setting up model %s", self.hparams.model.name)
            self.model = instantiate(
                self.hparams.model,
                input_shape=self.example_input_array.shape,
                labels=self.num_classes,
                _recursive_=False,
            )

        self._setup_loss_weights()

        # setup lists for reconstruction errors to compute anomaly threshold
        self.train_losses = list()
        self.normalized_train_errors = None
        self.predictions = torch.tensor([], device=self.device)
        self.labels = torch.tensor([], device=self.device)
        self.test_losses = list()
        self.encodings = dict()

        self.input_normalizer = BatchAugmentationPipeline(
            {"Normalize": {"mean": self.train_set.mean, "std": self.train_set.std}}
        )

        # Setup Augmentations
        self.default_augmentation = torch.nn.Identity()
        self.augmentations = {}

        self.setup_augmentations(self.hparams.augmentation)

        # Setup Metrics
        metrics = {}
        if self.num_classes > 0:
            self.test_confusion = ConfusionMatrix(
                "multiclass", num_classes=self.num_classes
            )

            for step_name in ["train", "val", "test"]:
                step_metrics = MetricCollection(
                    {
                        f"{step_name}_accuracy": Accuracy(
                            "multiclass", num_classes=self.num_classes
                        ),
                        f"{step_name}_error": Error(
                            "multiclass", num_classes=self.num_classes
                        ),
                        f"{step_name}_precision_micro": Precision(
                            "multiclass", num_classes=self.num_classes, average="micro"
                        ),
                        f"{step_name}_recall_micro": Recall(
                            "multiclass", num_classes=self.num_classes, average="micro"
                        ),
                        f"{step_name}_f1_micro": F1Score(
                            "multiclass", num_classes=self.num_classes, average="micro"
                        ),
                        f"{step_name}_precision_macro": Precision(
                            "multiclass", num_classes=self.num_classes, average="macro"
                        ),
                        f"{step_name}_recall_macro": Recall(
                            "multiclass", num_classes=self.num_classes, average="macro"
                        ),
                        f"{step_name}_f1_macro": F1Score(
                            "multiclass", num_classes=self.num_classes, average="macro"
                        ),
                    }
                )
                metrics[f"{step_name}_metrics"] = step_metrics

        self.metrics = torch.nn.ModuleDict(metrics)

        self.pseudo_label = instantiate(self.hparams.pseudo_labeling, model=self.model)

        # FIXME
        msglogger.info("Running dummy forward to initialize lazy modules")
        self.eval()
        self(self.example_input_array)
        self.train()

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

    def augment(self, images, labels, boxes, batch_idx, pipeline: Optional[str] = None):
        if boxes and (torch.numel(images) > 0):
            boxes_kornia = list()
            box_index = []
            for i in range(len(boxes)):
                if boxes[i]:  # not empty list
                    box = kornia.geometry.bbox.bbox_generator(
                        boxes[i][0][0], boxes[i][0][1], boxes[i][0][2], boxes[i][0][3]
                    )  # convert COCO to kornia format
                    boxes_kornia.append(box)
                    box_index.append(i)
            if not len(box_index) == 0:
                boxes_kornia = torch.cat(boxes_kornia)

        augmented_data = self.default_augmentation(images)
        if pipeline in self.augmentations:
            augmented_data = self.augmentations[pipeline].forward(augmented_data)

        elif pipeline is not None:
            msglogger.critical(
                "Could not find augmentations for `%s`, only default augmentations will be applied ",
                pipeline,
            )

        augmented_norm_data = self.input_normalizer.forward(augmented_data)

        if batch_idx == 0:
            pipeline_name = pipeline if pipeline is not None else "default"
            self._log_batch_images(
                f"augmented_{pipeline_name}", batch_idx, augmented_norm_data
            )

        return augmented_norm_data, images

    def setup_augmentations(self, pipeline_configs):
        default_augment = []
        augmentations = defaultdict(list)

        if pipeline_configs is None:
            msglogger.warning(
                "No data augmentations have been defined, make sure that this is intentional"
            )
            self.default_augmentation = torch.nn.Identity()
            return

        for pipeline_id, pipeline_config in pipeline_configs.items():
            pipeline_name = pipeline_config.get("pipeline", None)
            pipeline_transforms = BatchAugmentationPipeline(pipeline_config.transforms)

            if pipeline_name:
                augmentations[pipeline_name].append(pipeline_transforms)
            else:
                default_augment.append(pipeline_transforms)

        self.default_augmentation = torch.nn.Sequential(*default_augment)
        augmentations = {k: torch.nn.Sequential(*v) for k, v in augmentations.items()}
        self.augmentations = torch.nn.ModuleDict(augmentations)
