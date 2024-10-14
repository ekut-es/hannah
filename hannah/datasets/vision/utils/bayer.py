#
# Copyright (c) 2024 Hannah contributors.
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
import numpy as np


def rgb_to_bayer(image, pattern="RGGB", **params):
    """
    Convert an RGB image to a Bayer pattern.
    Args:
        image (np.ndarray): The input RGB image.
        pattern (str): The Bayer pattern to use. Can be one of 'RGGB', 'GBRG', 'GRBG', 'BGGR'.
    """

    bayer_image = np.zeros((image.shape[0], image.shape[1], 1), dtype=image.dtype)
    if pattern == "RGGB":  # Assuming the Bayer pattern is RGGB
        bayer_image[0::2, 0::2, 0] = image[0::2, 0::2, 0]  # Red
        bayer_image[1::2, 0::2, 0] = image[1::2, 0::2, 1]  # Green on Red row
        bayer_image[0::2, 1::2, 0] = image[0::2, 1::2, 1]  # Green on Blue row
        bayer_image[1::2, 1::2, 0] = image[1::2, 1::2, 2]  # Blue
    elif pattern == "GBRG":
        bayer_image[0::2, 0::2, 0] = image[0::2, 0::2, 1]  # Green on Blue row
        bayer_image[1::2, 0::2, 0] = image[1::2, 0::2, 0]  # Red
        bayer_image[0::2, 1::2, 0] = image[0::2, 1::2, 2]  # Blue
        bayer_image[1::2, 1::2, 0] = image[1::2, 1::2, 1]  # Green on Red row
    elif pattern == "GRBG":
        bayer_image[0::2, 0::2, 0] = image[0::2, 0::2, 1]  # Green on Red row
        bayer_image[1::2, 0::2, 0] = image[1::2, 0::2, 2]  # Blue
        bayer_image[0::2, 1::2, 0] = image[0::2, 1::2, 0]  # Red
        bayer_image[1::2, 1::2, 0] = image[1::2, 1::2, 1]  # Green on Blue row
    elif pattern == "BGGR":
        bayer_image[0::2, 0::2, 0] = image[0::2, 0::2, 2]  # Blue
        bayer_image[1::2, 0::2, 0] = image[1::2, 0::2, 1]  # Green on Blue row
        bayer_image[0::2, 1::2, 0] = image[0::2, 1::2, 1]  # Green on Red row
        bayer_image[1::2, 1::2, 0] = image[1::2, 1::2, 0]  # Red
    elif pattern is None:
        return image
    else:
        raise ValueError("Unknown Bayer pattern: {}".format(pattern))

    return bayer_image
