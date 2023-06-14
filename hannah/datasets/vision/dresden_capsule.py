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
"""Rhode island gastroenterology video capsule endoscopy dataset


    https://www.nature.com/articles/s41597-022-01726-3
    https://github.com/acharoen/Rhode-Island-GI-VCE-Technical-Validation
"""

import logging
import pathlib
import shutil

import numpy as np
import pandas as pd
import torchvision
import tqdm

from .base import ImageDatasetBase

logger = logging.getLogger(__name__)


def prepare_data(study_folder: pathlib.Path, data: pd.DataFrame):
    label_names = list(data.columns)[:-1]

    files = [study_folder / image for image in data["path"].to_list()]
    labels = np.argmax(data.iloc[:, :-1].values, axis=1)
    labels = [label_names[x] for x in labels]

    return files, labels, label_names


class DresdenCapsuleDataset(ImageDatasetBase):
    @classmethod
    def prepare(cls, config):
        pass

    @classmethod
    def splits(cls, config):
        data_folder = pathlib.Path(config["data_folder"]) / "dresden_capsule"
        study_folder = data_folder / "images"
        split_folder = data_folder / "splits" / config.task

        test_data = pd.read_csv(split_folder / "test.csv")
        val_data = pd.read_csv(split_folder / config.split / "val.csv")
        train_data = pd.read_csv(split_folder / config.split / "train.csv")

        X_train, y_train, labels = prepare_data(study_folder, train_data)
        X_val, y_val, labels = prepare_data(study_folder, val_data)
        X_test, y_test, labels = prepare_data(study_folder, test_data)

        train_set = cls(X_train, y_train, labels)
        val_set = cls(X_val, y_val, labels)
        test_set = cls(X_test, y_test, labels)

        # RANDOM, RANDOM_PER_STUDY Splits
        # preprocessing,

        return (
            train_set,
            val_set,
            test_set,
        )
