#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
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

import logging

try:
    import kornia.augmentation as A
except ModuleNotFoundError:
    logging.warning(
        "Could not import kornia augmentations if needed install with -E vision"
    )
    A = None

from .registry import registry

# Intensity Transformations
if A:
    registry.register(A.RandomMotionBlur)
    registry.register(A.ColorJiggle)
    registry.register(A.RandomBoxBlur)
    registry.register(A.RandomChannelShuffle)
    registry.register(A.RandomEqualize)
    registry.register(A.RandomGrayscale)
    registry.register(A.RandomGaussianBlur)
    registry.register(A.RandomGaussianNoise)
    registry.register(A.RandomMotionBlur)
    registry.register(A.RandomPosterize)
    registry.register(A.RandomSharpness)
    registry.register(A.RandomSolarize)

    # Color Transformations
    registry.register(A.CenterCrop)
    registry.register(A.RandomAffine)
    registry.register(A.RandomCrop)
    registry.register(A.RandomErasing)
    registry.register(A.RandomElasticTransform)
    registry.register(A.RandomFisheye)
    registry.register(A.RandomHorizontalFlip)
    registry.register(A.RandomInvert)
    registry.register(A.RandomPerspective)
    registry.register(A.RandomResizedCrop)
    registry.register(A.RandomRotation)
    registry.register(A.RandomVerticalFlip)
    registry.register(A.RandomThinPlateSpline)
