import logging
import os
import pathlib
import tarfile
from collections import Counter, namedtuple
from typing import Dict, List

import cv2
import pandas as pd
import requests
import torchvision
from albumentations.pytorch import ToTensorV2

import albumentations as A
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

        # FIXME(gerum):  add back rand augment
        train_transform = A.Compose(
            [
                A.RandomResizedCrop(config.resolution[0], config.resolution[1]),
                A.Normalize(mean=config.normalize.mean, std=config.normalize.std),
                ToTensorV2(),
            ]
        )

        test_transform = A.Compose(
            [
                A.Resize(config.resolution[0], config.resolution[1]),
                A.Normalize(mean=config.normalize.mean, std=config.normalize.std),
                ToTensorV2(),
            ]
        )

        label_to_folder = {
            "Angiectasia": "Angiectasia",
            "Pylorus": "Pylorus",
            "Blood": "Blood",
            "Reduced Mucosal View": "Reduced Mucosal View",
            "Ileo-cecal valve": "Ileo-cecal valve",
            "Erythematous": "Erythematous",
            "Foreign Bodies": "Foreign Bodies",
            "Normal": "Normal",
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
        else:
            raise Exception(
                f"Split {config.split} is not defined for dataset kvasir_capsule"
            )

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
        csv = pd.read_csv(self.csv_file)
        classes = [self.class_to_idx[csv.iloc[i][1]] for i in self.indices]
        counter = Counter(classes)
        classes_tupels = list(dict(counter).items())
        dataset_classes = list(self.class_to_idx.values())
        # random splits sometimes returns a val_split_Subset, that has 0 sampels from a dataset class, fill this with class index and None
        for i in dataset_classes:
            if i not in [j[0] for j in classes_tupels]:
                classes_tupels.append((i, None))
        counts = dict(sorted(classes_tupels))
        return counts

    @property
    def num_classes(self):
        return len(self.class_counts)

    # retuns a list of class index for every sample
    @property
    def get_label_list(self) -> List[int]:
        csv = pd.read_csv(self.csv_file)
        labels = [self.class_to_idx[csv.iloc[i][1]] for i in self.indices]
        return labels

    @property
    def class_names_abbreviated(self) -> List[str]:
        return [cn[0:3] for cn in self.class_names]
