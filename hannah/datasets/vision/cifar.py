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

import torch.utils.data as data
import torchvision
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import datasets

import albumentations as A

from .base import TorchvisionDatasetBase

logger = logging.getLogger(__name__)


class Cifar10Dataset(TorchvisionDatasetBase):
    @classmethod
    def prepare(cls, config):
        data_folder = config.data_folder
        root_folder = os.path.join(data_folder, "CIFAR10")
        _ = datasets.CIFAR10(root_folder, train=False, download=True)

    @property
    def class_names(self):
        q = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        return q

    @classmethod
    def splits(cls, config):
        data_folder = config.data_folder
        root_folder = os.path.join(data_folder, "CIFAR10")

        # print(loaded_transform)
        train_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=32),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomCrop(height=32, width=32),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                ToTensorV2(),
            ]
        )

        val_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=32),
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                ToTensorV2(),
            ]
        )

        test_set = torchvision.datasets.CIFAR10(
            root_folder, train=False, download=False
        )
        train_val_set = torchvision.datasets.CIFAR10(
            root_folder, train=True, download=False
        )
        train_val_len = len(train_val_set)

        split_sizes = [
            int(train_val_len * (1.0 - config.val_percent)),
            int(train_val_len * config.val_percent),
        ]
        train_set, val_set = data.random_split(train_val_set, split_sizes)

        return (
            cls(config, train_set, train_transform),
            cls(config, val_set, val_transform),
            cls(config, test_set, val_transform),
        )
