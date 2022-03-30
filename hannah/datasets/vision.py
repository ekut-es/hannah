import bisect
import collections
import json
import logging
import os
import pathlib
import tarfile
from collections import Counter
from logging import config
from posixpath import split
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
from hydra.utils import get_original_cwd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import albumentations as A
from hannah.modules.augmentation import rand_augment

from .base import AbstractDataset
from .utils import cachify, csv_dataset, generate_file_md5

try:
    import gdown
except ModuleNotFoundError:
    gdown = None


logger = logging.getLogger(__name__)


class VisionDatasetBase(AbstractDataset):
    """Wrapper around torchvision classification datasets"""

    def __init__(self, config, dataset, transform=None):
        self.config = config
        self.dataset = dataset
        self.transform = transform

    @property
    def class_counts(self):
        return None

    def __getitem__(self, index):
        data, target = self.dataset[index]
        data = np.array(data)
        if self.transform:
            data = self.transform(image=data)["image"]
        return data, target

    def size(self):
        dim = self[0][0].size()

        return list(dim)

    def __len__(self):
        return len(self.dataset)


class Cifar10Dataset(VisionDatasetBase):
    @classmethod
    def prepare(cls, config):
        data_folder = config.data_folder
        root_folder = os.path.join(data_folder, "CIFAR10")
        _ = datasets.CIFAR10(root_folder, train=False, download=True)

    @property
    def class_names(self):
        q = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        return q

    @classmethod
    def splits(cls, config):
        data_folder = config.data_folder
        root_folder = os.path.join(data_folder, "CIFAR10")

        # print(loaded_transform)
        train_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=32),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomCrop(height=32, width=32),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                ToTensorV2(),
            ]
        )

        val_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=32),
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                ToTensorV2(),
            ]
        )

        test_set = datasets.CIFAR10(root_folder, train=False, download=False)
        train_val_set = datasets.CIFAR10(root_folder, train=True, download=False)
        train_val_len = len(train_val_set)

        split_sizes = [
            int(train_val_len * (1.0 - config.val_percent)),
            int(train_val_len * config.val_percent),
        ]
        train_set, val_set = data.random_split(train_val_set, split_sizes)

        return (
            cls(config, train_set, train_transform),
            cls(config, val_set, val_transform),
            cls(config, test_set, val_transform),
        )


class FakeDataset(VisionDatasetBase):
    @classmethod
    def prepare(cls, config):
        pass

    @classmethod
    def splits(cls, config):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        test_data = torchvision.datasets.FakeData(
            size=128,
            image_size=(3, 32, 32),
            num_classes=config.num_classes,
            transform=transform,
        )
        val_data = torchvision.datasets.FakeData(
            size=128,
            image_size=(3, 32, 32),
            num_classes=config.num_classes,
            transform=transform,
        )
        train_data = torchvision.datasets.FakeData(
            size=512,
            image_size=(3, 32, 32),
            num_classes=config.num_classes,
            transform=transform,
        )

        return cls(config, train_data), cls(config, val_data), cls(config, test_data)

    @property
    def class_names(self):
        return [f"class{n}" for n in range(self.config.num_classes)]


class KvasirCapsuleDataset(VisionDatasetBase):
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

        # download and extract dataset
        if not os.path.isdir(extract_root):
            datasets.utils.download_and_extract_archive(
                cls.DOWNLOAD_URL,
                download_folder,
                extract_root=extract_root,
                filename="labelled_images.zip",
            )

            for tar_file in pathlib.Path(extract_root).glob("*.tar.gz"):
                # ampulla , hematin and polyp removed from official_splits due to small findings
                if str(tar_file) in [
                    extract_root + "/ampulla_of_vater.tar.gz",
                    extract_root + "/blood_hematin.tar.gz",
                    extract_root + "/polyp.tar.gz",
                ]:
                    tar_file.unlink()
                    continue
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
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(90),
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
            int(test_val_len * config.test_percent),
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


class KvasirCapsuleUnlabeled(AbstractDataset):
    """Dataset representing unalbelled videos"""

    BASE_URL = "https://files.osf.io/v1/resources/dv2ag/providers/googledrive/unlabelled_videos/"

    def __init__(self, config, metadata, transform=None):
        self.config = config
        self.metadata = metadata
        self.transform = transform

        self.data_root = (
            pathlib.Path(config.data_folder) / "kvasir_capsule" / "unlabelled_videos"
        )

        self.total_frames = 0
        # End frame of each video file when concatenated
        self.end_frames = []
        for video_data in metadata:
            self.total_frames += int(video_data["total_frames"])
            self.end_frames.append(self.total_frames)

        self._video_captures = {}

    @property
    def class_counts(self):
        return None

    @property
    def class_names(self):
        return []

    @property
    def class_names_abbreviated(self) -> List[str]:
        return []

    def _decode_frame(self, index):
        video_index = bisect.bisect_left(self.end_frames, index)
        assert video_index < len(self.metadata)

        video_metadata = dict(self.metadata[video_index])

        video_file = self.data_root / video_metadata["video_file"]
        assert video_file.exists()

        if video_file in self._video_captures:
            video_capture = self._video_captures[video_file]
        else:
            video_capture = cv2.VideoCapture(str(video_file))
            self._video_captures[video_file] = video_capture

        start_frame = 0
        if video_index > 0:
            start_frame = self.end_frames[video_index - 1]

        frame_index = index - start_frame

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        logger.debug("=============")
        logger.debug(video_index, start_frame, video_index, frame_index)

        ret, frame = video_capture.read()

        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = Image.fromarray(frame)

        return frame, video_metadata

    def __getitem__(self, index):
        res = {}

        data, metadata = self._decode_frame(index)

        if self.transform:
            data = self.transform(data)
        else:
            data = transforms.Resize(256)(data)
            data = transforms.CenterCrop(256)(data)
            data = transforms.Resize(224)(data)
            data = torchvision.transforms.ToTensor()(data)
            data = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(data)

        res["data"] = data
        res["metadata"] = metadata

        return res

    def size(self):
        dim = self[0][0].size()

        return list(dim)

    def __len__(self):
        return self.total_frames

    @classmethod
    def splits(cls, config):
        data_root = (
            pathlib.Path(config.data_folder) / "kvasir_capsule" / "unlabelled_videos"
        )
        files_json = data_root / "unlabelled_videos.json"
        assert files_json.exists()

        with files_json.open("r") as f:
            json_data = json.load(f)

        metadata = json_data["metadata"]
        train_split = metadata[:-2]
        val_split = [metadata[-2]]
        test_split = [metadata[-1]]

        train_set = cls(config, train_split)
        test_set = cls(config, test_split)
        val_set = cls(config, val_split)

        logger.debug("Train Data: %f Frames", train_set.total_frames)
        logger.debug("Val Data: %f Frames", val_set.total_frames)
        logger.debug("Test Data: %f Frames", test_set.total_frames)

        return train_set, val_set, test_set

    @classmethod
    def prepare(cls, config):

        data_root = (
            pathlib.Path(config.data_folder) / "kvasir_capsule" / "unlabelled_videos"
        )
        data_root.mkdir(parents=True, exist_ok=True)

        files_json = data_root / "unlabelled_videos.json"

        # download and extract dataset
        if not files_json.exists():
            logger.info("Getting file list from %s", cls.BASE_URL)
            with urllib.request.urlopen(cls.BASE_URL) as url:
                data = url.read()
                with files_json.open("w") as f:
                    f.write(data.decode())

        with files_json.open("r") as f:
            json_data = json.load(f)

        file_list = json_data["data"]

        video_metadata = []
        video_captures = []
        sum_frames = 0

        for file_data in tqdm(file_list, desc="Preparing dataset"):
            target_filename = data_root / file_data["attributes"]["name"]
            download_url = file_data["links"]["download"]
            expected_md5 = file_data["attributes"]["extra"]["hashes"]["md5"]

            needs_download = False
            if target_filename.exists():
                if "verify_files" in config and config.verify_files:
                    file_md5 = generate_file_md5(target_filename)
                    if not expected_md5 == file_md5:
                        logger.warning(
                            "MD5 of downloaded file does not match is: %s excpected: %s",
                            file_md5,
                            expected_md5,
                        )
                        needs_download = True
            else:
                needs_download = True

            if needs_download:
                if gdown is None:
                    logger.critical(
                        "Could not download %s due to missing package", target_filename
                    )
                    continue

                logger.info("downloading %s", target_filename)
                gdown.download(
                    url=download_url, output=str(target_filename), resume=True
                )

            if target_filename.suffix in [".mp4"]:
                video_capture = cv2.VideoCapture(str(target_filename))
                video_captures.append(video_capture)
                total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
                frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                frame_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                fps = video_capture.get(cv2.CAP_PROP_FPS)
                logger.debug("Total frames:     %f", total_frames)
                logger.debug("Frame height:     %f", frame_height)
                logger.debug("Frame width:      %f", frame_width)
                logger.debug("Frame rate (FPS): %f", fps)

                metadata = {
                    "video_file": target_filename.name,
                    "total_frames": total_frames,
                    "frame_height": frame_height,
                    "frame_width": frame_width,
                    "fps": fps,
                }

                video_metadata.append(metadata)

                sum_frames += total_frames
        logger.info("Sum of Frames total: %f", sum_frames)

        json_data["metadata"] = video_metadata
        with files_json.open("w") as f:
            json.dump(json_data, f)

        return None
