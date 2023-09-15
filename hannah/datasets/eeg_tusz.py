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
        self.X = dataset[set]["X"]
        self.y = dataset[set]["y"]
        # print(np.unique(dataset[set]["y"], return_counts=True))

        self._class_counts = dict(Counter(self.y))

    def __getitem__(self, index):
        x = self.X[index] / self.mean_value  # normalize
        data = torch.tensor(x)
        label = torch.tensor([self.y[index]]).long()
        return data, data.shape[0], label, label.shape[0]

    def __len__(self):
        return len(self.X)

    @classmethod
    def splits(cls, config):
        data_folder = Path(config.data_folder)
        split_name = config.split

        split_folder = data_folder / split_name

        train_set = EEGDataset(
            config,
            "/nfs/wsi/es/hannah/datasets/tuh_seizure_detection/preprocessed/common_channels_simple/train_subset_tusz_4s_ar_with_normal_before.hdf5",
            "train",
        )
        val_set = EEGDataset(
            config,
            "/nfs/wsi/es/hannah/datasets/tuh_seizure_detection/preprocessed/common_channels_simple/dev_subset_tusz_4s_ar_with_normal.hdf5",
            "dev",
        )
        test_set = EEGDataset(
            config,
            "/nfs/wsi/es/hannah/datasets/tuh_seizure_detection/preprocessed/common_channels_simple/eval_subset_tusz_4s_ar_with_normal.hdf5",
            "eval",
        )
        mean_value = np.mean(test_set.X)

        def mean_normalization(data_set, mean_value):
            for i in range(data_set.X.shape[0]):
                if i < data_set.X.shape[0] / 100000:
                    mean_value = (
                        mean_value + np.mean(data_set.X[100000 * i : 100000 * (i + 1),])
                    ) / 2
                else:
                    break
            return mean_value

        mean_value = mean_normalization(train_set, mean_value)
        mean_value = mean_normalization(val_set, mean_value)
        print("Mean value for normalization of all datasets is: ", mean_value)
        train_set.mean_value = mean_value
        val_set.mean_value = mean_value
        test_set.mean_value = mean_value
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
    def get_label_list(self) -> List[int]:
        return self.y
