import bisect
import json
import logging
import pathlib
import urllib
from typing import List

import cv2
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm

import albumentations as A

from ..base import AbstractDataset
from ..utils import generate_file_md5

try:
    import gdown
except ModuleNotFoundError:
    gdown = None


logger = logging.getLogger(__name__)


class KvasirCapsuleUnlabeled(AbstractDataset):
    """Dataset representing unlabelled videos"""

    BASE_URL_UNLABELED = "https://files.osf.io/v1/resources/dv2ag/providers/googledrive/unlabelled_videos/"
    BASE_URL_LABELED = (
        "https://files.osf.io/v1/resources/dv2ag/providers/googledrive/labelled_videos/"
    )

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
            self.end_frames.append(self.total_frames - 1)

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

        start_frame = 0
        if video_index > 0:
            start_frame = self.end_frames[video_index - 1] + 1

        frame_index = min(
            max(index - start_frame, 0), video_metadata["total_frames"] - 1
        )

        if video_file in self._video_captures:
            video_capture = self._video_captures[video_file]
        else:
            video_capture = cv2.VideoCapture(str(video_file))
            self._video_captures[video_file] = video_capture

        ret, frame = video_capture.read()
        if not ret:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = video_capture.read()
            assert ret == True

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, video_metadata

    def __getitem__(self, index):
        res = {}

        # start_time = time.time()
        data, metadata = self._decode_frame(index)
        # end_time = time.time()

        # print("Decode  time", end_time - start_time)

        augmented = self.transform(image=data)
        data = augmented["image"]

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

        mean = tuple(config.mean)
        std = tuple(config.mean)
        resolution = tuple(config.resolution)

        train_transforms = A.Compose(
            [
                # A.RandomResizedCrop(height=config.resolution[0], width=config.resolution[1], scale=(0.5,1.0)),
                A.Resize(height=resolution[0], width=resolution[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=180, p=1.0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            p=1.0,
        )

        test_transforms = A.Compose(
            [
                A.Resize(height=resolution[0], width=resolution[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            p=1.0,
        )

        train_set = cls(config, train_split, transform=train_transforms)
        test_set = cls(config, test_split, transform=test_transforms)
        val_set = cls(config, val_split, transform=test_transforms)

        logger.debug("Train Data: %f Frames", train_set.total_frames)
        logger.debug("Val Data: %f Frames", val_set.total_frames)
        logger.debug("Test Data: %f Frames", test_set.total_frames)

        return train_set, val_set, test_set

    @classmethod
    def prepare(cls, config):
        cls._prepare_unlabled(config)
        cls._prepare_labeled(config)

    @classmethod
    def _prepare_labeled(cls, config):
        pass

    @classmethod
    def _prepare_unlabled(cls, config):
        data_root = (
            pathlib.Path(config.data_folder) / "kvasir_capsule" / "unlabelled_videos"
        )
        data_root.mkdir(parents=True, exist_ok=True)

        files_json = data_root / "unlabelled_videos.json"

        # download and extract dataset
        if not files_json.exists():
            logger.info("Getting file list from %s", cls.BASE_URL)
            with urllib.request.urlopen(cls.BASE_URL_UNLABELLED) as url:
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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_video_captures"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._video_captures = {}
