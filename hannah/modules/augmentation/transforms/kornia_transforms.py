#
# Copyright (c) 2023 Hannah contributors.
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

import torchvision

try:
    import kornia.augmentation as K
    import kornia.geometry.transform
except ModuleNotFoundError:
    logging.warning(
        "Could not import kornia augmentations if needed install with -E vision"
    )
    K = None

from .registry import registry

# Intensity Transformations
if K:
    registry.register(K.RandomMotionBlur)
    registry.register(K.ColorJiggle)
    registry.register(K.RandomBoxBlur)
    registry.register(K.RandomChannelShuffle)
    registry.register(K.RandomEqualize)
    registry.register(K.RandomGrayscale)
    registry.register(K.RandomGaussianBlur)
    registry.register(K.RandomGaussianNoise)
    registry.register(K.RandomMotionBlur)
    registry.register(K.RandomPosterize)
    registry.register(K.RandomSharpness)
    registry.register(K.RandomSolarize)
    registry.register(K.Normalize)

    # Color Transformations
    registry.register(K.CenterCrop)
    registry.register(K.RandomAffine)
    registry.register(K.RandomCrop)
    registry.register(K.RandomErasing)
    registry.register(K.RandomElasticTransform)
    registry.register(K.RandomFisheye)
    registry.register(K.RandomHorizontalFlip)
    registry.register(K.RandomInvert)
    registry.register(K.RandomPerspective)
    registry.register(K.RandomResizedCrop)
    registry.register(K.RandomRotation)
    registry.register(K.RandomVerticalFlip)
    registry.register(K.RandomThinPlateSpline)
