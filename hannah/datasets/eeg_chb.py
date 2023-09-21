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
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch

from .base import AbstractDataset


class EEGDataset(AbstractDataset):
    def __init__(self, config, file, set):
        super().__init__()
        self.config = config
        dataset = h5py.File(file, "r")
        self.X = dataset[set]["x"]
        self.y = dataset[set]["y"]
        # print(np.unique(dataset[set]["y"], return_counts=True))

        self._class_counts = {
            i: k for i, k in enumerate(np.unique(self.y, return_counts=True)[1])
        }

    def __getitem__(self, index):
        x = self.X[index] / self.max_value  # normalize
        data = torch.tensor(x)
        label = torch.tensor([self.y[index]]).long()

        return data, data.shape[0], label, label.shape[0]

    def __len__(self):
        return len(self.X)

    @classmethod
    def splits(cls, config):
        dataset_name = config.get("dataset_name", None)
        if dataset_name is None:
            raise AttributeError(
                "Please provide the dataset name in /home/kohlibha/exploration/data/---"
            )

        train_set = EEGDataset(
            config,
            f"{config.data_folder}/{config.dataset}/preprocessed/{dataset_name}/dev/full.hdf5",
            "train",
        )
        val_set = EEGDataset(
            config,
            f"{config.data_folder}/{config.dataset}/preprocessed/{dataset_name}/dev/full.hdf5",
            "val",
        )
        test_set = EEGDataset(
            config,
            f"{config.data_folder}/{config.dataset}/preprocessed/{dataset_name}/test/full.hdf5",
            "test",
        )
        min_max_train = max(np.abs(np.min(train_set.X)), np.max(train_set.X))
        min_max_val = max(np.abs(np.min(val_set.X)), np.max(val_set.X))
        min_max_test = max(np.abs(np.min(test_set.X)), np.max(test_set.X))
        min_max_all = max(min_max_test, min_max_train, min_max_val)
        train_set.max_value = min_max_all
        val_set.max_value = min_max_all
        test_set.max_value = min_max_all

        return train_set, val_set, test_set

    @classmethod
    def prepare(cls, config):
        pass

    @property
    def class_names(self) -> List[str]:
        """Returns the names of the classes in the classification dataset"""
        return ["background", "seizure"]

    @property
    def class_counts(self) -> Optional[Dict[int, int]]:
        """Returns the number of items in each class of the dataset

        If this is not applicable to a dataset type e.g. ASR, Semantic Segmentation,
        it may return None
        """

        return self._class_counts

    def size(self) -> List[int]:
        """Returns dimension of output without batch dimension"""

        return list(self.X[0].shape)

    @property
    def label_list(self) -> List[int]:
        return self.y
