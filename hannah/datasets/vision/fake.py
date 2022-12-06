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

import torchvision

from .base import TorchvisionDatasetBase

logger = logging.getLogger(__name__)

import albumentations as A
from albumentations.pytorch import ToTensorV2


class FakeDataset(TorchvisionDatasetBase):

    resolution = 320

    @classmethod
    def prepare(cls, config):
        pass

    @classmethod
    def splits(cls, config):
        transform = (
            None  # torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        )

        test_data = torchvision.datasets.FakeData(
            size=128,
            image_size=(3, cls.resolution, cls.resolution),
            num_classes=config.num_classes,
            transform=transform,
        )
        val_data = torchvision.datasets.FakeData(
            size=128,
            image_size=(3, cls.resolution, cls.resolution),
            num_classes=config.num_classes,
            transform=transform,
        )
        train_data = torchvision.datasets.FakeData(
            size=512,
            image_size=(3, cls.resolution, cls.resolution),
            num_classes=config.num_classes,
            transform=transform,
        )

        return cls(config, train_data), cls(config, val_data), cls(config, test_data)

    @property
    def class_names(self):
        return [f"class{n}" for n in range(self.config.num_classes)]
