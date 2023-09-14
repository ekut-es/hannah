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
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torchvision

from ..utils.utils import extract_from_download_cache, list_all_files
from .base import AbstractDataset, DatasetType
from .vision.base import TorchvisionDatasetBase


class Fake1dDataset(TorchvisionDatasetBase):
    @classmethod
    def prepare(cls, config):
        pass

    @classmethod
    def splits(cls, config):
        resolution = config.resolution
        channels = config.channels

        test_data = torchvision.datasets.FakeData(
            size=128,
            image_size=(channels, resolution),
            num_classes=config.num_classes,
        )
        val_data = torchvision.datasets.FakeData(
            size=128,
            image_size=(channels, resolution),
            num_classes=config.num_classes,
        )
        train_data = torchvision.datasets.FakeData(
            size=512,
            image_size=(channels, resolution),
            num_classes=config.num_classes,
        )

        return cls(config, train_data), cls(config, val_data), cls(config, test_data)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        data = np.array(data).astype(np.float32) / 255
        data = self.transform(image=data)["image"]
        data = torch.squeeze(data)
        return data, target

    @property
    def class_names(self):
        return [f"class{n}" for n in range(self.config.num_classes)]
