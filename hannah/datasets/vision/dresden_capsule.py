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

import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
import albumentations as A
import tqdm
from sklearn.utils import resample
from .base import ImageDatasetBase

logger = logging.getLogger(__name__)


def prepare_data(study_folder: pathlib.Path, data: pd.DataFrame):

    label_names = list(data.columns)[:-1]
    files = [study_folder / image for image in data["path"].to_list()]

    if len(label_names) > 1: # True for section and technical tasks
        labels = np.argmax(data.iloc[:, :-1].values, axis=1)
    
    else: # Assuming binary task (some anomaly and normal)
        label_names.insert(0, 'normal')
        labels = np.max(data.iloc[:, :-1].values, axis=1)
    
    labels = [label_names[x] for x in labels]

    return files, labels, label_names

def downsampling(X: list, y: list, labels: list, config: dict):
    
    task = config.task
    seed = config.seed

    if task == 'sections' or task == 'technical_multiclass_view' or task == 'technical_multilabel_bubbles_dirt':
        idx_resampled = np.empty(0, dtype=int)
        for i in range(len(labels)):
            y = np.array(y)
            idx_y = np.where(y == labels[i])[0] # get only one class
            n_samples = int(config.downsampling.ratio[task][i]*len(idx_y))
            idx_resampled_temp = resample(idx_y, n_samples=n_samples, random_state=seed) # downsample class to n samples
            idx_resampled = np.concatenate([idx_resampled, idx_resampled_temp])

    else: # Assuming binary task
        ratio = config.downsampling.ratio.binary
        y = np.array(y)
        normal_idx  = np.where(y == labels[0])[0]
        anomaly_idx = np.where(y == labels[1])[0]
        n_samples = int(len(anomaly_idx)*config.downsampling.anomalies_fraction)
        idx_resampled_anomaly = resample(anomaly_idx, n_samples=n_samples, random_state=seed)
        idx_resampled_normal = resample(normal_idx, n_samples=int(n_samples*ratio), random_state=seed)
        idx_resampled = np.concatenate([idx_resampled_anomaly, idx_resampled_normal])

    ordered_idx_resampled = np.sort(idx_resampled)
    y = y[ordered_idx_resampled]
    X = np.array(X)[ordered_idx_resampled]
    assert len(X) == len(y)

    return X, y


class DresdenCapsuleDataset(ImageDatasetBase):
    @classmethod
    def prepare(cls, config):
        pass

    @classmethod
    def splits(cls, config):
        data_folder = pathlib.Path(config["data_folder"]) / "dresden-capsule"
        study_folder = data_folder / "images"
        split_folder = data_folder / "splits_tuebingen" / config.task

        test_data = pd.read_csv(split_folder / "test.csv")
        val_data = pd.read_csv(split_folder / config.split / "val.csv")
        train_data = pd.read_csv(split_folder / config.split / "train.csv")

        X_train, y_train, labels = prepare_data(study_folder, train_data)
        X_val, y_val, labels = prepare_data(study_folder, val_data)
        X_test, y_test, labels = prepare_data(study_folder, test_data)

        # Resampling
        if config.downsampling.enabled:
            X_train, y_train = downsampling(X_train, y_train, labels, config)
            X_val, y_val = downsampling(X_val, y_val, labels, config)

        transform = A.Compose([A.augmentations.geometric.resize.Resize(config.sensor.resolution[0], config.sensor.resolution[1]), ToTensorV2()])
        test_transform = A.Compose([A.augmentations.geometric.resize.Resize(config.sensor.resolution[0], config.sensor.resolution[1]), ToTensorV2()])
        train_set = cls(X_train, y_train, labels, transform=transform)
        val_set = cls(X_val, y_val, labels, transform=test_transform)
        test_set = cls(X_test, y_test, labels, transform=test_transform)

        return (
            train_set,
            val_set,
            test_set,
        )
