import os
import csv

import numpy as np
from torch.utils.data import Dataset

import torch
import torch.nn.functional as F

from torchvision import transforms

from pl_bolts.utils import _PIL_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

# from .util import KittiLabel
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

        self.label_names = config["labels"]
        self.img_size = tuple(map(int, config["img_size"].split(",")))
        self.kitti_dir = config["kitti_folder"]
        self.img_path = os.path.join(self.kitti_dir, self.IMAGE_PATH)
        self.label_path = os.path.join(self.kitti_dir, self.LABEL_PATH)
        self.set_type = set_type
        self.img_files = list(data.keys())
        self.label_files = list(data.values())
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        pil_img = Image.open(self.img_path + self.img_files[idx]).convert("RGB")
        # pil_img = pil_img.resize(self.img_size)
        pil_img = self.transform(pil_img)

        target = {}
        label = self._parse_label(idx)

        labels = []
        boxes = []

        for i in range(len(label)):
            boxes.append(torch.Tensor(label[i].get("bbox")))
            labels.append(torch.tensor(label[i].get("type"), dtype=torch.long))

        target["boxes"] = torch.stack(boxes)
        target["labels"] = torch.stack(labels)

        return pil_img, target

    def _parse_label(self, idx: int):
        label = []
        with open(self.label_path + self.label_files[idx]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                label.append(
                    {
                        "type": self.label_names.get(line[0]),
                        "truncated": float(line[1]),
                        "occluded": int(line[2]),
                        "alpha": float(line[3]),
                        "bbox": [float(x) for x in line[4:8]],
                        "dimensions": [float(x) for x in line[8:11]],
                        "location": [float(x) for x in line[11:14]],
                        "rotation_y": float(line[14]),
                    }
                )
        return label

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


def object_collate_fn(data):
    return tuple(zip(*data))
    """
    def merge(sequences):
        lengths = [seq.shape[-1] for seq in sequences]
        max_length = max(lengths)

        padded_seqs = []

        for item in sequences:
            padded = torch.nn.functional.pad(
                input=item,
                pad=(0, max_length - item.shape[-1]),
                mode="constant",
                value=0,
            )
            padded_seqs.append(padded)

        return padded_seqs, lengths

    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    # trg_seqs, trg_lengths = merge(trg_seqs)

    return (torch.stack(src_seqs), trg_seqs)
    """
