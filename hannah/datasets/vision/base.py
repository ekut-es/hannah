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
import logging
import re
from collections import Counter
from typing import List

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

from ..base import AbstractDataset

logger = logging.getLogger(__name__)


class VisionDatasetBase(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self._resolution = [224, 224]

    @property
    def std(self):
        return (0.5, 0.5, 0.5)

    @property
    def mean(self):
        return (0.5, 0.5, 0.5)

    @property
    def resolution(self):
        return self._resolution


class TorchvisionDatasetBase(VisionDatasetBase):
    """Wrapper around torchvision classification datasets"""

    def __init__(self, config, dataset, transform=None):
        super().__init__(config)
        self.dataset = dataset
        self.transform = transform if transform else A.Compose([ToTensorV2()])

    @property
    def class_counts(self):
        return None

    def __getitem__(self, index):
        data, target = self.dataset[index]
        data = np.array(data).astype(np.float32) / 255
        data = self.transform(image=data)["image"]
        return data, target

    def size(self):
        dim = self[0][0].shape
        return tuple(dim)

    def __len__(self):
        return len(self.dataset)


class ImageDatasetBase(VisionDatasetBase):
    def __init__(self, X, y, classes, bbox=None, transform=None, metadata=None):
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
        self.metadata = metadata

    def __getitem__(self, index):
        id_study = ""
        image = cv2.imread(str(self.X[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        label = self.y[index]
        bbox = []
        metadata = {}

        X_filename = re.search(r"([^\/]+).$", str(self.X[index]))[0]
        if self.metadata: 
            metadata = {key: self.metadata[key][index] for key in self.metadata}
        if (
            self.bbox and X_filename in self.bbox
        ):  # bounding box for anomaly detection tasks
            bbox = self.bbox[X_filename]

        data = self.transform(image=image)["image"]
        target = self.label_to_index[label]
        return {"data": data, "labels": target, "bbox": bbox, "metadata": metadata}

    def size(self):
        dim = self[0]["data"].shape
        return list(dim)

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
        res = {}
        for label in self.classes:
            i = self.label_to_index[label]
            if label in counts:
                res[i] = counts[label]
            else:
                res[i] = 0
        return res

    @property
    def num_classes(self):
        return len(self.classes)

    # retuns a list of class index for every sample
    @property
    def label_list(self) -> List[int]:
        return [self.label_to_index[y] for y in self.y]

    @property
    def class_names_abbreviated(self) -> List[str]:
        return [cn[0:3] for cn in self.class_names]
