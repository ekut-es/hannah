#
# Copyright (c) 2023 Hannah contributors.
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
import json
import logging
import os
import pathlib
import re
import tarfile
from collections import Counter, defaultdict, namedtuple

import albumentations as A
import numpy as np
import pandas as pd
import requests
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from .base import ImageDatasetBase

logger = logging.getLogger(__name__)


class KvasirCapsuleDataset(ImageDatasetBase):
    DOWNLOAD_URL = "https://files.osf.io/v1/resources/dv2ag/providers/googledrive/labelled_images/?zip="
    METADATA_JSON_URL = (
        "https://files.osf.io/v1/resources/dv2ag/providers/googledrive/metadata.json"
    )
    METADATA_CSV_URL = "https://osf.io/download/kzc8w/"

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
        metadata_json_path = os.path.join(
            config.data_folder, "kvasir_capsule", "metadata.json"
        )
        if not os.path.exists(metadata_json_path):
            metadata_request = requests.get(cls.METADATA_JSON_URL)
            with open(metadata_json_path, "wb") as f:
                f.write(metadata_request.content)

        metadata_csv_path = os.path.join(
            config.data_folder, "kvasir_capsule", "metadata.csv"
        )
        if not os.path.exists(metadata_csv_path):
            metadata_request = requests.get(cls.METADATA_CSV_URL)
            with open(metadata_csv_path, "wb") as f:
                f.write(metadata_request.content)

    @classmethod
    def splits(cls, config):

        data_root = os.path.join(
            config.data_folder, "kvasir_capsule", "labelled_images"
        )
        split_root = os.path.join(
            config.data_folder, "kvasir_capsule", "official_splits"
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

        metadata_path = os.path.join(
            config.data_folder, "kvasir_capsule", "metadata.csv"
        )
        metadata = (
            pd.read_csv(metadata_path, sep=";").dropna(axis=0).astype(pd.StringDtype())
        )

        def process_bbox(paths):
            bbox = defaultdict(list)
            for path in paths:
                X_filename = re.search(r"([^\/]+).$", path)[0]  # get filename of jpg
                if (metadata["filename"] == X_filename).any():
                    rows = metadata[
                        metadata["filename"].str.match(X_filename)
                    ]  # rows of interest

                    for index, row in rows.iterrows():
                        x_min = np.min(
                            row[["x1", "x2", "x3", "x4"]].to_numpy(dtype=np.float32)
                        )
                        y_min = np.min(
                            row[["y1", "y2", "y3", "y4"]].to_numpy(dtype=np.float32)
                        )
                        x_max = np.max(
                            row[["x1", "x2", "x3", "x4"]].to_numpy(dtype=np.float32)
                        )
                        y_max = np.max(
                            row[["y1", "y2", "y3", "y4"]].to_numpy(dtype=np.float32)
                        )
                        width = x_max - x_min
                        height = y_max - y_min
                        single_bbox = torch.from_numpy(
                            np.array([x_min, y_min, width, height])
                        )  # COCO format

                        bbox[X_filename].append(single_bbox)
            return bbox

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

            split0_bbox = process_bbox(split0_paths)
            split1_bbox = process_bbox(split1_paths)

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

        transform = A.Compose([A.augmentations.geometric.resize.Resize(config.sensor.resolution[0], config.sensor.resolution[1]), ToTensorV2()])
        return (
            cls(
                train_images,
                train_labels,
                classes,
                split0_bbox,
                transform=transform,
            ),
            cls(
                val_images,
                val_labels,
                classes,
                split0_bbox,
            ),
            cls(
                test_images,
                test_labels,
                classes,
                split1_bbox,
            ),
        )
