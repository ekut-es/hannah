import os
import csv
import numpy as np
from torch.utils.data import Dataset

import torch
import torch.nn.functional as F

import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

from pycocotools.coco import COCO

from torchvision import transforms

from pl_bolts.utils import _PIL_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

from .base import DatasetType, AbstractDataset

if _PIL_AVAILABLE:
    from PIL import Image
else:  # pragma: no cover
    warn_missing_pkg("PIL", pypi_name="Pillow")


class Kitti(AbstractDataset):
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
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.28679871, 0.30261545, 0.32524435],
                    std=[0.27106311, 0.27234113, 0.27918578],
                ),
            ]
        )
        self.cocoGt = KittiCOCO(
            self.img_files,
            self.img_size,
            self.label_names,
            self.img_path,
            self.kitti_dir,
        )

    @classmethod
    def prepare(cls, config):
        pass

    @property
    def class_names(self):
        return list(self.label_names.keys())

    @property
    def class_counts(self):
        return None

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

        for la in label:
            boxes.append(torch.Tensor(la.get("bbox")))
            labels.append(torch.tensor(la.get("type"), dtype=torch.long))
            self.cocoGt.addAnn(idx, la.get("type"), la.get("bbox"))

        target["boxes"] = torch.stack(boxes)
        target["labels"] = torch.stack(labels)
        target["filename"] = self.img_files[idx]

        return pil_img, target

    def getCocoGt(self):
        return self.cocoGt

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
        num_imgs = 7479
        num_test_imgs = math.floor(num_imgs * (config["test_pct"] / 100))
        num_dev_imgs = math.floor(num_imgs * (config["dev_pct"] / 100))

        datasets = [{}, {}, {}]

        if num_imgs > 7480:
            raise Exception("Number of images for Kitti dataset too large")
        elif num_test_imgs < 1 or num_dev_imgs < 1:
            raise Exception("Each step must have at least 1 Kitti image")

        for i in range(num_imgs):
            # first test_pct into test dataset
            if i < num_test_imgs:
                datasets[0][str(i).zfill(6) + ".png"] = str(i).zfill(6) + ".txt"
            # next images into dev dataset
            elif i < num_test_imgs + num_dev_imgs:
                datasets[1][str(i).zfill(6) + ".png"] = str(i).zfill(6) + ".txt"
            # last images into train dataset
            else:
                datasets[2][str(i).zfill(6) + ".png"] = str(i).zfill(6) + ".txt"

        res_datasets = (
            cls(datasets[2], DatasetType.TRAIN, config),
            cls(datasets[1], DatasetType.DEV, config),
            cls(datasets[0], DatasetType.TEST, config),
        )

        return res_datasets


class KittiCOCO(COCO):
    def __init__(self, img_files, img_size, label_names, img_path, kitti_dir):
        super().__init__()
        self.img_path = img_path
        self.kitti_dir = kitti_dir

        dataset = dict()
        dataset["images"] = []
        dataset["categories"] = []
        dataset["annotations"] = []

        i = 0
        for img in img_files:
            img_dict = dict()
            img_dict["id"] = i
            img_dict["width"] = img_size[1]
            img_dict["height"] = img_size[0]
            img_dict["filename"] = img
            dataset["images"].append(img_dict)
            i += 1

        for label, no in zip(label_names, range(len(label_names))):
            label_dict = dict()
            label_dict["id"] = no
            label_dict["name"] = label
            dataset["categories"].append(label_dict)

        self.dataset = dataset

    def addAnn(self, idx, catId, bbox):
        ann_dict = dict()
        coco_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        ann_dict["id"] = len(self.dataset["annotations"]) + 1
        ann_dict["image_id"] = idx
        ann_dict["category_id"] = catId
        ann_dict["segmentation"] = "polygon"
        ann_dict["bbox"] = coco_bbox
        ann_dict["iscrowd"] = 0
        ann_dict["area"] = coco_bbox[2] * coco_bbox[3]

        self.dataset["annotations"].append(ann_dict)

    def clearBatch(self):
        self.dataset["annotations"] = []

    def _getImgId(self, filename):
        for img in range(len(self.imgs)):
            if self.imgs[img]["filename"] == filename:
                return self.imgs[img]["id"]

    def transformOutput(self, output, y):
        retval = []

        for boxes, labels, scores, y_img in zip(
            (out["boxes"] for out in output),
            (out["labels"] for out in output),
            (out["scores"] for out in output),
            y,
        ):
            for box, label, score in zip(boxes, labels, scores):
                img_dict = dict()
                x1 = box[0].item()
                y1 = box[1].item()
                img_dict["image_id"] = self._getImgId(y_img["filename"])
                img_dict["category_id"] = label.item()
                img_dict["bbox"] = [x1, y1, box[2].item() - x1, box[3].item() - y1]
                img_dict["score"] = score.item()
                retval.append(img_dict)

        if len(retval) == 0:
            return COCO()
        return self.loadRes(retval)

    def saveImg(self, cocoDt, y):

        for y_img in y:
            filename = y_img["filename"]
            cocoImg = self._getImgId(filename)
            img = mpimg.imread(self.img_path + filename)
            fig, ax = plt.subplots()
            ax.imshow(img)

            annsGt = self.loadAnns(self.getAnnIds(imgIds=cocoImg))
            annsDt = cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=cocoImg))

            if annsDt == []:
                print("")

            for ann in annsGt:
                # if ann["category_id"] != 0:
                box = ann["bbox"]
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2],
                    box[3],
                    linewidth=1,
                    edgecolor="b",
                    facecolor="none",
                )
                ax.add_patch(rect)

            for ann in annsDt:
                box = ann["bbox"]
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2],
                    box[3],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)

            if not os.path.exists("./ann"):
                os.makedirs("./ann")

            plt.savefig("./ann/" + filename)
            plt.close()


def object_collate_fn(data):
    return tuple(zip(*data))
