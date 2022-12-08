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
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DatasetType(Enum):
    """The type of a dataset partition e.g. train, dev, test"""

    TRAIN = 0
    DEV = 1
    TEST = 2


class AbstractDataset(Dataset, ABC):
    @abstractclassmethod
    def prepare(cls, config: Dict[str, Any]) -> None:
        """Prepare the dataset.

        This method is run at the beginning of the dataset training.

        If possible it should download the dataset from its original source, if
        it is available for public download.


        Args:
            config (Dict[Any]): The dataset configuration
        """

        pass

    @abstractclassmethod
    def splits(
        cls, config: Dict[str, Any]
    ) -> Tuple["AbstractDataset", "AbstractDataset", "AbstractDataset"]:
        """Returns the test, validation and train split according to the Dataset config

        Args:
            config ([type]): [description]
        """

        pass  # pytype: disable=bad-return-type

    @abstractproperty
    def class_names(self) -> List[str]:
        """Returns the names of the classes in the classification dataset"""
        pass  # pytype: disable=bad-return-type

    @property
    def class_names_abbreviated(self) -> List[str]:
        max_len = 0
        for name in self.class_names:
            max_len = max(max_len, len(name))
        if max_len > 6:
            logger.warning(
                "Datasets class names contain classes that are longer than the recommended lenght of 5 characters, consider implementing class_names_abbreviated in your dataset"
            )

        return self.class_names

    @abstractproperty
    def class_counts(self) -> Optional[Dict[int, int]]:
        """Returns the number of items in each class of the dataset

        If this is not applicable to a dataset type e.g. ASR, Semantic Segementation,
        it may return None
        """
        pass

    @abstractmethod
    def __getitem__(self, index) -> List[torch.Tensor]:
        """Returns a torch.Tensor for the data item at the corresponding index

        The length of the list depends on the dataset item to use

        Args:
            index (int): the index of the data item
        """

        pass  # pytype: disable=bad-return-type

    @abstractmethod
    def __len__(self) -> int:
        """Returns number of samples in dataset"""

        pass  # pytype: disable=bad-return-type

    def size(self) -> List[int]:
        """Returns dimension of output output without batch dimension"""

        return [self.channels, self.input_length]

    @property
    def std(self) -> Optional[Tuple[int, ...]]:
        """Returns channel-wise standard deviation for dataset if applicable"""
        return None

    @property
    def mean(self) -> Optional[Tuple[int, ...]]:
        """Returns channel-wise means for dataset if applicable"""
        return None

    @property
    def weights(self) -> Optional[List[float]]:
        """Class weights for weighted sampling"""
        class_counts = self.class_counts
        if class_counts:
            counts = list(class_counts.values())
            weights = [1 / i for i in counts]
            return weights
