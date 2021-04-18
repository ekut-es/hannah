import os
import random

import numpy as np
from torch.utils.data import Dataset

from pl_bolts.utils import _PIL_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

from .base import DatasetType

if _PIL_AVAILABLE:
    from PIL import Image
else:  # pragma: no cover
    warn_missing_pkg("PIL", pypi_name="Pillow")


class Kitti(Dataset):
    ""

    IMAGE_PATH = os.path.join("training", "image_2/")

    LABEL_PATH = os.path.join("training", "label_2/")

    def __init__(self, data, set_type, config):
        if not _PIL_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use `PIL` which is not installed yet."
            )

        self.img_size = tuple(map(int, config["img_size"].split(",")))
        self.kitti_dir = config["kitti_folder"]
        self.img_path = os.path.join(self.kitti_dir, self.IMAGE_PATH)
        self.label_path = os.path.join(self.kitti_dir, self.LABEL_PATH)
        self.set_type = set_type
        self.img_files = list(data.keys())
        self.img_labels = list(data.values())

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_path + self.img_files[idx])
        img = img.resize(self.img_size)
        img = np.array(img)

        return img

    @classmethod
    def splits(cls, config):
        """Splits the dataset in training, devlopment and test set and returns
        the three sets as List"""

        folder = config["data_folder"]
        folder = os.path.join(folder, "kitti")

        descriptions = ["train.txt", "dev.txt", "test.txt"]
        datasets = [{}, {}, {}]

        for num, desc in enumerate(descriptions):
            descs = os.path.join(folder, desc)
            f = open(descs, "r")

            for line in f:
                file_name = line.rstrip("\n")
                datasets[num][line.rstrip("\n") + ".png"] = file_name + ".txt"
            f.close

        res_datasets = (
            cls(datasets[0], DatasetType.TRAIN, config),
            cls(datasets[1], DatasetType.DEV, config),
            cls(datasets[2], DatasetType.TEST, config),
        )

        return res_datasets

    @classmethod
    def download(cls, config):
        return
