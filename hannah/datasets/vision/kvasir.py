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
import pathlib
import tarfile
from collections import Counter, namedtuple
from typing import Dict, List

import albumentations as A
import cv2
import pandas as pd
import requests
import torchvision
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from hannah.modules.augmentation import rand_augment

from ..base import AbstractDataset

logger = logging.getLogger(__name__)


class KvasirCapsuleDataset(AbstractDataset):
    DOWNLOAD_URL = "https://files.osf.io/v1/resources/dv2ag/providers/googledrive/labelled_images/?zip="
    METADATA_URL = (
        "https://files.osf.io/v1/resources/dv2ag/providers/googledrive/metadata.json"
    )

    def __init__(self, config, X, y, classes, transform=None):
        self.data_root = (
            pathlib.Path(config.data_folder) / "kvasir_capsule" / "labelled_images"
        )

        self.X = X
        self.y = y
        self.classes = classes
        self.transform = transform
        self.label_to_index = {k: v for v, k in enumerate(classes)}

    def __getitem__(self, index):
        image = cv2.imread(str(self.X[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.y[index]
        if self.transform:
            data = self.transform(image=image)["image"]
        target = self.label_to_index[label]
        return {"data": data, "labels": target}

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.X)

    @classmethod
    def prepare(cls, config):
        download_folder = os.path.join(config.data_folder, "downloads")
        extract_root = os.path.join(
            config.data_folder, "kvasir_capsule", "labelled_images"
        )
        # download and extract dataset
        if not os.path.isdir(extract_root):
            torchvision.datasets.utils.download_and_extract_archive(
                cls.DOWNLOAD_URL,
                download_folder,
                extract_root=extract_root,
                filename="labelled_images.zip",
            )

            for tar_file in pathlib.Path(extract_root).glob("*.tar.gz"):
                logger.info("Extracting: %s", str(tar_file))
                with tarfile.open(tar_file) as archive:
                    archive.extractall(path=extract_root)
                tar_file.unlink()

        # download splits
        official_splits_path = os.path.join(
            config.data_folder, "kvasir_capsule", "official_splits"
        )
        if not os.path.isdir(official_splits_path):
            os.mkdir(official_splits_path)
            s0 = requests.get(
                "https://raw.githubusercontent.com/simula/kvasir-capsule/master/official_splits/split_0.csv"
            )
            with open(os.path.join(official_splits_path, "split_0.csv"), "wb") as f:
                f.write(s0.content)
            s1 = requests.get(
                "https://raw.githubusercontent.com/simula/kvasir-capsule/master/official_splits/split_1.csv"
            )
            with open(os.path.join(official_splits_path, "split_1.csv"), "wb") as f:
                f.write(s1.content)

        # download metadata
        metadata_path = os.path.join(
            config.data_folder, "kvasir_capsule", "metadata.json"
        )
        if not os.path.exists(metadata_path):
            metadata_request = requests.get(cls.METADATA_URL)
            with open(metadata_path, "wb") as f:
                f.write(metadata_request.content)

    @classmethod
    def splits(cls, config):
        data_root = os.path.join(
            config.data_folder, "kvasir_capsule", "labelled_images"
        )
        split_root = os.path.join(
            config.data_folder, "kvasir_capsule", "official_splits"
        )

        resolution = config.resolution
        if isinstance(resolution, int):
            resolution = (resolution, resolution)

        res_x, res_y = resolution

        # FIXME(gerum):  add back rand augment
        train_transform = A.Compose(
            [
                A.RandomResizedCrop(res_x, res_y),
                A.Normalize(mean=config.normalize.mean, std=config.normalize.std),
                ToTensorV2(),
            ]
        )

        test_transform = A.Compose(
            [
                A.Resize(res_x, res_y),
                A.Normalize(mean=config.normalize.mean, std=config.normalize.std),
                ToTensorV2(),
            ]
        )

        label_to_folder = {
            "Angiectasia": "Angiectasia",
            "Pylorus": "Pylorus",
            "Blood": "Blood - fresh",
            "Reduced Mucosal View": "Reduced mucosal view",
            "Ileo-cecal valve": "Ileocecal valve",
            "Erythematous": "Erythema",
            "Foreign Bodies": "Foreign body",
            "Normal": "Normal clean mucosa",
            "Ulcer": "Ulcer",
            "Erosion": "Erosion",
            "Lymphangiectasia": "Lymphangiectasia",
            # Not Used in official splits dataset
            "Ampulla of vater": "Ampulla of vater",
            "Polyp": "Polyp",
            "Blood - hematin": "Blood - hematin",
        }

        if config.split == "official":

            def process_official_split(df: pd.DataFrame):

                files = df["filename"].to_list()
                labels = df["label"].to_list()

                folders = [label_to_folder[label] for label in labels]

                paths = [
                    os.path.join(data_root, folder, file)
                    for folder, file in zip(folders, files)
                ]

                return paths, labels

            split0 = pd.read_csv(os.path.join(split_root, "split_0.csv"))
            split1 = pd.read_csv(os.path.join(split_root, "split_1.csv"))

            split0_paths, split0_labels = process_official_split(split0)
            split1_paths, split1_labels = process_official_split(split1)

            train_images = split0_paths
            train_labels = split0_labels

            val_images = split0_paths
            val_labels = split0_labels

            test_images = split1_paths
            test_labels = split1_labels

            classes = list(set(split0_labels + split1_labels))
            classes.sort()
        elif config.split == "random":
            images = []
            labels = []

            for label, folder in label_to_folder.items():
                current_folder = pathlib.Path(data_root) / folder
                for file in current_folder.glob("*.jpg"):
                    images.append(file)
                    labels.append(label)
            classes = list(set(labels))
            classes.sort()

            train_images, test_images, train_labels, test_labels = train_test_split(
                images, labels, test_size=0.2, random_state=1
            )
            train_images, val_images, train_labels, val_labels = train_test_split(
                train_images, train_labels, test_size=0.25, random_state=1
            )
        else:
            raise Exception(
                f"Split {config.split} is not defined for dataset kvasir_capsule"
            )

        if config.get("anomaly"):
            classes = ["Normal", "Anomaly"]

            def relable_anomaly(X):
                label_to_anomaly = {
                    "Angiectasia": "Anomaly",
                    "Pylorus": "Normal",
                    "Blood": "Anomaly",
                    "Reduced Mucosal View": "Anomaly",
                    "Ileo-cecal valve": "Normal",
                    "Erythematous": "Anomaly",
                    "Foreign Bodies": "Anomaly",
                    "Normal": "Normal",
                    "Ulcer": "Anomaly",
                    "Erosion": "Anomaly",
                    "Lymphangiectasia": "Anomaly",
                    # Not Used in official splits dataset
                    "Ampulla of vater": "Normal",
                    "Polyp": "Anomaly",
                    "Blood - hematin": "Anomaly",
                }

                return [label_to_anomaly[x] for x in X]

            train_labels = relable_anomaly(train_labels)
            val_labels = relable_anomaly(val_labels)
            test_labels = relable_anomaly(test_labels)

        return (
            cls(
                config,
                train_images,
                train_labels,
                classes,
                transform=train_transform,
            ),
            cls(
                config,
                val_images,
                val_labels,
                classes,
                transform=test_transform,
            ),
            cls(
                config,
                test_images,
                test_labels,
                classes,
                transform=test_transform,
            ),
        )

    @property
    def class_names(self):
        return self.classes

    @property
    def class_counts(self):
        counter = Counter(self.y)
        counts = dict(counter)
        for i in len(self.classes):
            if i not in counts:
                counts[i] = 0
        return counts

    @property
    def num_classes(self):
        return len(self.class_counts)

    # retuns a list of class index for every sample
    @property
    def get_label_list(self) -> List[int]:
        return self.y

    @property
    def class_names_abbreviated(self) -> List[str]:
        return [cn[0:3] for cn in self.class_names]
