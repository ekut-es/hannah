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
import random
from collections import defaultdict

import numpy as np
import torch
import torchvision

from ..utils import extract_from_download_cache, list_all_files
from .base import AbstractDataset, DatasetType


class Fake1dDataset(AbstractDataset):
    def __init__(self, config, size):
        self.config = config
        self.size = size

        self.data = torch.randn((size, config["channels"], config["resolution"])).split(
            1, 0
        )
        self.target = torch.randn(
            (size, config.size), dtype=torch.int32, min=0, max=config["num_classes"]
        )

    @classmethod
    def prepare(cls, config):
        pass

    @classmethod
    def splits(cls, config):
        return cls(config, size=128), cls(config, size=32), cls(config, size=32)

    def __getitem__(self, index):
        data, target = self.data[index], self.target[index]
        return data, target

    @property
    def class_names(self):
        return [f"class{n}" for n in range(self.config.num_classes)]

    def __len__(self):
        return len(self.targets)
