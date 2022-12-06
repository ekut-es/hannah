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
import torch

from hannah.datasets.collate import vision_collate_fn


def test_vision_collate_fn():
    bbox = [[], [torch.rand(4), torch.rand(4)], [], [torch.rand(4)]]
    batch = [{"bbox": boxes} for boxes in bbox]

    collated = vision_collate_fn(batch)

    bbox_collated = collated["bbox"]

    assert bbox_collated[0] == []
    assert bbox_collated[2] == []

    print(bbox_collated)

    # assert bbox_collated[1] == []
    # assert bbox_collated[3] == []


def test_vision_collate_fn2():
    bbox = [[], [torch.rand(2, 4)], [], [torch.rand(1, 4)]]
    batch = [{"bbox": boxes} for boxes in bbox]

    collated = vision_collate_fn(batch)

    bbox_collated = collated["bbox"]

    assert bbox_collated[0] == []
    assert bbox_collated[2] == []

    print(bbox_collated)


def test_vision_collate_fn3():
    bbox = [[torch.rand(2, 4)], [], [torch.rand(1, 4)]]
    batch = [{"bbox": boxes} for boxes in bbox]

    collated = vision_collate_fn(batch)

    bbox_collated = collated["bbox"]

    assert bbox_collated[1] == []

    print(bbox_collated)
