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
import pathlib
import tarfile
from collections import Counter, namedtuple
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import requests
import torchvision
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from timm.data.mixup import Mixup

import albumentations as A
from hannah.modules.augmentation import rand_augment
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


class ImageDatasetBase(AbstractDataset):
    def __init__(self, X, y, classes, transform=None):
        """Initialize vision dataset

        Args:
            X (List[str]): List of paths to image files
            y (List[str]): Class id of corresponding image
            classes (List[str]): List of class names, names are ordered by numeric class id
            transform (Callable[image,image], optional): Optional transformation/augmentation of input images. Defaults to None.
        """
        self.X = X
        self.y = y
        self.classes = classes
        self.transform = (
            transform if transform else A.Compose([A.Normalize(), ToTensorV2()])
        )
        self.label_to_index = {k: v for v, k in enumerate(classes)}

    def __getitem__(self, index):
        image = cv2.imread(str(self.X[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.y[index]
        if self.transform:
            data = self.transform(image=image)["image"]
        else:
            data = image
        target = self.label_to_index[label]
        return {"data": data, "labels": target}

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.X)

    @property
    def class_names(self):
        return self.classes

    @property
    def class_counts(self):
        counter = Counter(self.y)
        counts = dict(counter)
        for i in len(self.classes):
            if i not in counts:
                counts[i] = 0
        return counts

    @property
    def num_classes(self):
        return len(self.class_counts)

    # retuns a list of class index for every sample
    @property
    def get_label_list(self) -> List[int]:
        return self.y

    @property
    def class_names_abbreviated(self) -> List[str]:
        return [cn[0:3] for cn in self.class_names]
