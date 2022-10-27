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

import numpy as np
from timm.data.mixup import Mixup

from hannah.modules.augmentation.batch_augmentation import BatchAugmentationPipeline

from ..base import AbstractDataset

logger = logging.getLogger(__name__)


class AugmentationMixin:
    def get_mixup_fn(self):
        mixup = None

        augmentation_config = self.config.get("augmentation", None)
        if augmentation_config is not None:
            mixup_config = augmentation_config.get("mixup", None)
            if mixup_config is not None:
                mixup_config = Mixup(**mixup_config)

        return mixup

    def get_batch_augment_fn(self):

        batch_augment = None

        augmentation_config = self.config.get("augmentation", None)
        if augmentation_config is not None:
            batch_augment_config = self.config.get("batch_augment", None)
            if batch_augment_config is not None:

                batch_augment = BatchAugmentationPipeline(
                    transforms=batch_augment_config.get("batch_transforms", [])
                )

        return batch_augment


class VisionDatasetBase(AbstractDataset, AugmentationMixin):
    def __init__(self, config):
        self.config = config


class TorchvisionDatasetBase(VisionDatasetBase):
    """Wrapper around torchvision classification datasets"""

    def __init__(self, config, dataset, transform=None):
        super().__init__(config)
        self.dataset = dataset
        self.transform = transform

    @property
    def class_counts(self):
        return None

    def __getitem__(self, index):
        data, target = self.dataset[index]
        data = np.array(data)
        if self.transform:
            data = self.transform(image=data)["image"]
        return data, target

    def size(self):
        dim = self[0][0].size()

        return list(dim)

    def __len__(self):
        return len(self.dataset)
