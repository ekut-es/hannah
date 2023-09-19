#
# Copyright (c) 2023 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
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
import bisect
import concurrent.futures
import json
import logging
import pathlib
import urllib
from typing import List

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm

from ..base import AbstractDataset
from ..utils import generate_file_md5

try:
    import gdown
except ModuleNotFoundError:
    gdown = None


logger = logging.getLogger(__name__)


def _decode_frame(index, metadata, end_frames, data_root):
    video_index = bisect.bisect_left(end_frames, index)
    assert video_index < len(metadata)

    video_metadata = dict(metadata[video_index])

    video_file = data_root / video_metadata["video_file"]
    assert video_file.exists()

    start_frame = 0
    if video_index > 0:
        start_frame = end_frames[video_index - 1] + 1

    frame_index = min(max(index - start_frame, 0), video_metadata["total_frames"] - 1)

    video_capture = cv2.VideoCapture(str(video_file))

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = video_capture.read()
    assert ret is True

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return frame, video_metadata


class KvasirCapsuleUnlabeled(AbstractDataset):
    """Dataset representing unlabelled videos"""

    BASE_URL_UNLABELED = "https://files.osf.io/v1/resources/dv2ag/providers/googledrive/unlabelled_videos/"
    BASE_URL_LABELED = (
        "https://files.osf.io/v1/resources/dv2ag/providers/googledrive/labelled_videos/"
    )

    def __init__(self, config, metadata, transform=None):
        self.config = config
        self.metadata = metadata
        self.transform = transform if transform else A.Compose([ToTensorV2()])

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

        # Cache Config
        self.poolsize = 2048
        self.queuesize = 1
        self.workqueue = []
        self.data_pool = []
        self.worker_pool = concurrent.futures.ProcessPoolExecutor()

        self._update_cache()

    def _update_cache(self):
        done_index = []
        for index, fut in enumerate(self.workqueue):
            if fut.done():
                self.data_pool.append(fut.result())

                done_index.append(index)

        for index in sorted(done_index, reverse=True):
            if index < len(self.workqueue):
                del self.workqueue[index]

        if len(self.data_pool) > self.poolsize:
            start = len(self.data_pool) - self.poolsize
            self.data_pool = self.data_pool[start:]

        for _ in range(self.queuesize - len(self.workqueue)):
            next_number = np.random.randint(low=0, high=self.total_frames)
            self.workqueue.append(
                self.worker_pool.submit(
                    _decode_frame,
                    next_number,
                    self.metadata,
                    self.end_frames,
                    self.data_root,
                )
            )

    @property
    def class_counts(self):
        return None

    @property
    def class_names(self):
        return []

    @property
    def class_names_abbreviated(self) -> List[str]:
        return []

    def __getitem__(self, index):
        res = {}

        # start_time = time.time()

        self._update_cache()
        while len(self.data_pool) < self.queuesize:
            self._update_cache()

        index = np.random.randint(low=0, high=len(self.data_pool))
        data, metadata = self.data_pool[index]

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
        cls.prepare(config)
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

        transform = A.Compose(
            [
                A.augmentations.geometric.resize.Resize(
                    config.sensor.resolution[0], config.sensor.resolution[1]
                ),
                ToTensorV2(),
            ]
        )
        train_set = cls(config, train_split, transform=transform)
        test_set = cls(config, test_split)
        val_set = cls(config, val_split)

        logger.debug("Train Data: %f Frames", train_set.total_frames)
        # logger.debug("Val Data: %f Frames", val_set.total_frames)
        # logger.debug("Test Data: %f Frames", test_set.total_frames)

        return train_set, None, None

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
            logger.info("Getting file list from %s", cls.BASE_URL_UNLABELED)
            with urllib.request.urlopen(cls.BASE_URL_UNLABELED) as url:
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
                # video_captures.append(video_capture)
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
        if True:  # not files_json.exists:
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

    @property
    def sequential(self) -> bool:
        """Returns true if this dataset should only be iterated sequentially"""

        return True

    @property
    def max_workers(self) -> int:
        """Returns the maximum number of workers useable for this dataset"""

        return 0
