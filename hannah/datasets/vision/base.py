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
import re
import tarfile
from collections import Counter, namedtuple
from typing import Dict, List

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from timm.data.mixup import Mixup

from hannah.modules.augmentation import rand_augment
from hannah.modules.augmentation.batch_augmentation import BatchAugmentationPipeline

from ..base import AbstractDataset

logger = logging.getLogger(__name__)


class VisionDatasetBase(AbstractDataset):
    def __init__(self, config):
        self.config = config

    @property
    def std(self):
        pass

    @property
    def mean(self):
        pass


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
        data = np.array(data) / 255
        if self.transform:
            data = self.transform(image=data)["image"]
        return data, target

    def size(self):
        dim = self[0][0].size()

        return list(dim)

    def __len__(self):
        return len(self.dataset)


class ImageDatasetBase(AbstractDataset):
    def __init__(self, X, y, classes, bbox=None, transform=None):
        """Initialize vision dataset

        Args:
            X (List[str]): List of paths to image files
            y (List[str]): Class id of corresponding image
            classes (List[str]): List of class names, names are ordered by numeric class id
            bbox (Dict[str]): Dict with filename as keys, bbox coordinates as numpy arrays
            transform (Callable[image,image], optional): Optional transformation/augmentation of input images. Defaults to None.
        """
        self.X = X
        self.y = y
        self.classes = classes
        self.transform = transform if transform else A.Compose([ToTensorV2()])
        self.label_to_index = {k: v for v, k in enumerate(classes)}
        self.bbox = bbox

    def __getitem__(self, index):
        image = cv2.imread(str(self.X[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        label = self.y[index]

        bbox = []
        X_filename = re.search(r"([^\/]+).$", str(self.X[index]))[0]
        if self.bbox and X_filename in self.bbox:
            bbox = self.bbox[X_filename]
        data = self.transform(image=image)["image"]
        target = self.label_to_index[label]
        return {"data": data, "labels": target, "bbox": bbox}

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
