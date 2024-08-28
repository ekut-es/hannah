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
import os
import torchvision
from .base import TorchvisionDatasetBase


class SVHNDataset(TorchvisionDatasetBase):
    @classmethod
    def prepare(cls, config):
        data_folder = config.data_folder
        root_folder = os.path.join(data_folder, "SVHN")
        _ = torchvision.datasets.SVHN(root_folder, split="test", download=True)
        _ = torchvision.datasets.SVHN(root_folder, split="train", download=True)
        _ = torchvision.datasets.SVHN(root_folder, split="extra", download=True)

    @classmethod
    def splits(cls, config):
        data_folder = config.data_folder
        root_folder = os.path.join(data_folder, "SVHN")
        test_set = torchvision.datasets.SVHN(root_folder, split="test", download=False)
        train_set = torchvision.datasets.SVHN(
            root_folder, split="train", download=False
        )
        extra_set = torchvision.datasets.SVHN(
            root_folder, split="extra", download=False
        )

        return (cls(config, train_set), cls(config, test_set), cls(config, extra_set))

    @property
    def class_names(self):
        return [str(i) for i in range(10)]

    @property
    def std(self):
        return (0.5, 0.5, 0.5)

    @property
    def mean(self):
        return (0.5, 0.5, 0.5)
