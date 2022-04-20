import bisect
import json
import logging
import os
import pathlib
import tarfile
import urllib
from collections import Counter
from typing import Dict, List, Tuple

import cv2
import gdown
import pandas as pd
import requests
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from hannah.modules.augmentation import rand_augment

from ..base import AbstractDataset
from ..utils import generate_file_md5

logger = logging.getLogger(__name__)


class KvasirCapsuleDataset(AbstractDataset):
    def __init__(self, config, dataset, indices=None, csv_file=None, transform=None):
        super().__init__(config, dataset, transform)
        self.indices = indices
        self.csv_file = csv_file
        classes, class_to_idx = self.find_classes(config.data_root)
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        data, target = self.dataset[index]
        if self.transform:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    DOWNLOAD_URL = "https://files.osf.io/v1/resources/dv2ag/providers/googledrive/labelled_images/?zip="

    @classmethod
    def prepare(cls, config):
        download_folder = os.path.join(config.data_folder, "downloads")
        extract_root = os.path.join(
            config.data_folder, "kvasir_capsule", "labelled_images"
        )
        force = config.get("force")

        # download and extract dataset
        if not os.path.isdir(extract_root) or force == True:
            torchvision.datasets.utils.download_and_extract_archive(
                cls.DOWNLOAD_URL,
                download_folder,
                extract_root=extract_root,
                filename="labelled_images.zip",
            )

            for tar_file in pathlib.Path(extract_root).glob("*.tar.gz"):
                # ampulla , hematin and polyp removed from official_splits due to small findings
                logger.info("Extracting: %s", str(tar_file))
                with tarfile.open(tar_file) as archive:
                    archive.extractall(path=extract_root)
                tar_file.unlink()
            # rename dataset image labels to match official splits csv
            class_names = {
                "Angiectasia": "Angiectasia",
                "Erosion": "Erosion",
                "Pylorus": "Pylorus",
                "Blood - fresh": "Blood",
                "Erythema": "Erythematous",
                "Foreign body": "Foreign Bodies",
                "Ileocecal valve": "Ileo-cecal valve",
                "Lymphangiectasia": "Lymphangiectasia",
                "Normal clean mucosa": "Normal",
                "Pylorus": "Pylorus",
                "Reduced mucosal view": "Reduced Mucosal View",
                "Ulcer": "Ulcer",
            }
            for c_name in class_names:
                new_file = os.path.join(extract_root, class_names[c_name])
                old_file = os.path.join(extract_root, c_name)
                os.rename(old_file, new_file)

        # downlaod splits
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

    @classmethod
    def splits(cls, config):
        data_root = os.path.join(
            config.data_folder, "kvasir_capsule", "labelled_images"
        )

        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize(224),
                rand_augment.RandAugment(
                    config.augmentations.rand_augment.N,
                    config.augmentations.rand_augment.M,
                    config=config,  # FIXME: make properly configurable
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        test_transofrm = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        train_val_set = csv_dataset.DatasetCSV(
            config.train_split, data_root, transform=None
        )
        test_set = csv_dataset.DatasetCSV(
            config.test_val_split, data_root, transform=None
        )

        test_val_len = len(test_set)
        split_sizes = [
            int(test_val_len * (1.0 - config.val_percent)),
            int(test_val_len * config.val_percent),
        ]
        # # split_1 has odd number
        if test_val_len != split_sizes[0] + split_sizes[1]:
            split_sizes[0] = split_sizes[0] + 1

        test_val_splits, test_indices, val_indices = csv_dataset.random_split(
            test_set, split_sizes
        )
        test_set = test_val_splits[0]
        val_set = test_val_splits[1]
        train_indices = [i for i in range(len(train_val_set.imgs))]

        return (
            cls(
                config,
                train_val_set,
                train_indices,
                config.train_split,
                transform=train_transform,
            ),
            cls(
                config,
                val_set,
                val_indices,
                config.test_val_split,
                transform=test_transofrm,
            ),
            cls(
                config,
                test_set,
                test_indices,
                config.test_val_split,
                transform=test_transofrm,
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

    @property
    def weights(self):
        # FIXME: move to base classes
        counts = list(self.class_counts.values())
        weights = [1 / i for i in counts]
        return weights
