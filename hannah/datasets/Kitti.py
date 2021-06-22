import os
import sys
import csv
import glob
import numpy as np
from torch.functional import Tensor
from torch.utils.data import Dataset

import torch
import torch.nn.functional as F

import math

import shutil

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

try:
    from pycocotools.coco import COCO
except ModuleNotFoundError:
    COCO = None

from torchvision import transforms

from .base import DatasetType, AbstractDataset

from PIL import Image


class Kitti(AbstractDataset):
    ""

    IMAGE_PATH = os.path.join("training/image_2/")

    AUG_PATH = os.path.join("training/augmented_2/")

    LABEL_PATH = os.path.join("training", "label_2/")

    def __init__(self, data, set_type, config):
        if COCO is None:
            logging.error("Could not find pycocotools")
            logging.error(
                "please install with poetry install 'poetry install -E object-detection'"
            )
            sys.exit(-1)

        self.set_type = set_type
        self.label_names = config["labels"]
        self.img_size = tuple(map(int, config["img_size"].split(",")))
        self.kitti_dir = config["kitti_folder"]
        self.img_path = os.path.join(self.kitti_dir, self.IMAGE_PATH)
        self.aug_path = os.path.join(self.kitti_dir, self.AUG_PATH)
        self.label_path = os.path.join(self.kitti_dir, self.LABEL_PATH)
        self.img_files = list(data.keys())
        self.aug_files = list()
        self.label_files = list(data.values())
        self.labels_ignore = config["labels_ignore"]
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
            self.labels_ignore,
            self.img_path,
            self.aug_path,
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
        img_name = self.img_files[idx]

        if img_name[:-4] in self.aug_files and os.path.isfile(
            self.kitti_dir + "/training/augmented/" + img_name
        ):
            path = self.aug_path
            shutil.copy2(
                self.kitti_dir + "/training/augmented/" + img_name,
                self.aug_path + img_name,
            )
        else:
            path = self.img_path

        pil_img = Image.open(path + img_name).convert("RGB")
        # pil_img = pil_img.resize(self.img_size)
        pil_img = self.transform(pil_img)

        target = {}
        label = self._parse_label(
            idx, True if self.set_type == DatasetType.TRAIN else False
        )

        labels = []
        boxes = []

        for la in label:
            boxes.append(torch.Tensor(la.get("bbox")))
            labels.append(torch.tensor(la.get("type"), dtype=torch.long))
            self.cocoGt.addAnn(idx, la.get("type"), la.get("bbox"))

        target["boxes"] = torch.stack(boxes)
        target["labels"] = torch.stack(labels)
        target["filename"] = self.img_files[idx]
        target["augmented"] = True if path == self.aug_path else False

        return pil_img, target

    def getCocoGt(self):
        return self.cocoGt

    def _parse_label(self, idx: int, considerDC: bool):
        label = []
        with open(self.label_path + self.label_files[idx]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                if (
                    considerDC is False
                    or not self.label_names.get(line[0]) in self.labels_ignore
                ):
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

        folder = config["kitti_folder"]
        folder = os.path.join(folder, "training")
        aug_folder = os.path.join(folder, "augmented/")
        aug2_folder = os.path.join(folder, "augmented_2/")
        folder = os.path.join(folder, "image_2/")
        files = sorted(
            filter(
                lambda x: os.path.isfile(os.path.join(folder, x)), os.listdir(folder)
            )
        )
        num_imgs = len(files)
        num_test_imgs = math.floor(num_imgs * (config["test_pct"] / 100))
        num_dev_imgs = math.floor(num_imgs * (config["dev_pct"] / 100))

        datasets = [{}, {}, {}]

        if num_test_imgs < 1 or num_dev_imgs < 1:
            raise Exception("Each step must have at least 1 Kitti image")

        if "real_rain" not in folder:
            if os.path.exists(aug2_folder) and os.path.isdir(aug2_folder):
                shutil.rmtree(aug2_folder)
            os.mkdir(aug2_folder)

            if os.path.exists(aug_folder) and os.path.isdir(aug_folder):
                shutil.rmtree(aug_folder)
            os.mkdir(aug_folder)

        for i in range(num_imgs):
            # test_img pct into test dataset
            if i < num_test_imgs:
                img_name = files[i]
                datasets[0][img_name] = files[i][:-4] + ".txt"
            # dev_img pct into val dataset
            elif i < num_test_imgs + num_dev_imgs:
                img_name = files[i]
                datasets[1][img_name] = files[i][:-4] + ".txt"
            # last pictures into training set
            else:
                img_name = files[i]
                datasets[2][img_name] = (
                    files[i][:-4] + ".txt"
                )  # last imgs not augmented

        res_datasets = (
            cls(datasets[2], DatasetType.TRAIN, config),
            cls(datasets[1], DatasetType.DEV, config),
            cls(datasets[0], DatasetType.TEST, config),
        )

        return res_datasets


class KittiCOCO(COCO):
    labels_ignore = list()

    def __init__(
        self,
        img_files,
        img_size,
        label_names,
        labels_ignore,
        img_path,
        aug_path,
        kitti_dir,
    ):
        super().__init__()
        self.img_path = img_path
        self.aug_path = aug_path
        self.kitti_dir = kitti_dir
        KittiCOCO.labels_ignore = labels_ignore

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
        if catId not in KittiCOCO.labels_ignore:
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

    def getImgId(self, filename):
        for img in range(len(self.imgs)):
            if self.imgs[img]["filename"] == filename:
                return self.imgs[img]["id"]

    def saveImg(self, cocoDt, y):

        for y_img in y:
            path = self.aug_path if y_img["augmented"] else self.img_path
            filename = y_img["filename"]
            cocoImg = self.getImgId(filename)
            img = mpimg.imread(path + filename)
            fig, ax = plt.subplots()
            ax.imshow(img)

            annsGt = self.loadAnns(self.getAnnIds(imgIds=cocoImg))
            annsDt = cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=cocoImg))

            if annsDt == []:
                print("")

            for ann in annsGt:
                if self.cats[ann["category_id"]]["id"] not in KittiCOCO.labels_ignore:
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
                if self.cats[ann["category_id"]]["id"] not in KittiCOCO.labels_ignore:
                    box = ann["bbox"]
                    rect = patches.Rectangle(
                        (box[0], box[1]),
                        box[2],
                        box[3],
                        linewidth=1,
                        edgecolor="r",
                        facecolor="none",
                    )
                    ax.text(
                        box[0],
                        box[1],
                        self.cats[ann["category_id"]]["name"]
                        if ann["category_id"] in self.cats
                        else "undefined",
                        color="red",
                        fontsize=10,
                    )
                    ax.add_patch(rect)

            if not os.path.exists("./ann"):
                os.makedirs("./ann")

            plt.savefig("./ann/" + filename)
            plt.close()

    @staticmethod
    def dontCareMatch(box: Tensor, size, img: Tensor):
        for i in range(len(img["labels"])):
            if img["labels"][i] in KittiCOCO.labels_ignore:

                gt = np.zeros((size[0], size[1]))
                dt = np.zeros((size[0], size[1]))

                gt_x1 = int(img["boxes"][i][0])
                gt_y1 = int(img["boxes"][i][1])
                gt_x2 = int(img["boxes"][i][2])
                gt_y2 = int(img["boxes"][i][3])

                dt_x1 = int(box[0])
                dt_y1 = int(box[1])
                dt_x2 = int(box[2])
                dt_y2 = int(box[3])

                gt[gt_x1:gt_x2, gt_y1:gt_y2] = np.ones(
                    (
                        gt_x2 - gt_x1,
                        gt_y2 - gt_y1,
                    )
                )

                dt[
                    max(dt_x1, 0) : min(dt_x2, size[0]),
                    max(dt_y1, 0) : min(dt_y2, size[1]),
                ] = np.ones(
                    (
                        min(dt_x2, size[0]) - max(dt_x1, 0),
                        min(dt_y2, size[1]) - max(dt_y1, 0),
                    )
                )

                intersection = (np.logical_and(gt, dt)).sum()
                iou_score = intersection / dt.sum()
                if iou_score > 0.5:
                    return True

        return False


def object_collate_fn(data):
    return tuple(zip(*data))
