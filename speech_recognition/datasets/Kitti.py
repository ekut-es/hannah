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

    IMAGE_PATH = os.path.join("training", "image_2")

    def __init__(self, data, set_type, config):
        if not _PIL_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use `PIL` which is not installed yet."
            )

        self.img_size = config["img_size"]
        self.data_dir = config["img_folder"]
        self.img_path = os.path.join(self.data_dir, self.IMAGE_PATH)
        self.img_list = self.get_filenames(self.img_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = img.resize(self.img_size)
        img = np.array(img)

        return img

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list

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

            img_files = []

            for line in f:
                img_files.append(line.rstrip("\n") + ".png")
            f.close

            datasets[num] = img_files

        res_datasets = (
            cls(datasets[0], DatasetType.TRAIN, config),
            cls(datasets[1], DatasetType.DEV, config),
            cls(datasets[2], DatasetType.TEST, config),
        )

        return res_datasets

    @classmethod
    def download(cls, config):
        return
