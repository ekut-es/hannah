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
from typing import Any, Optional, TypeVar, Union

import torch
import torch.nn as nn


class FixedPointNormalizer(nn.Module):
    "Simple feature normalizer for fixed point models"

    def __init__(
        self,
        normalize_bits: int = 8,
        normalize_max: int = 256,
        divide=False,
        override_max=False,
    ) -> None:
        super().__init__()
        self.normalize_bits = normalize_bits
        self.normalize_max = normalize_max
        self.divide = divide
        self.bits = self.normalize_bits - 1

        if self.divide and self.normalize_bits % 2 == 0:

            self.bits = int((self.normalize_bits / 2) - 1)
            self.low_border = (2**self.bits) - 1
            self.high_border = self.low_border << self.bits

            if not override_max:
                self.normalize_max = self.high_border

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalize_factor = 2.0**self.bits
        x = x * normalize_factor / self.normalize_max
        x = x.round()

        if self.divide:

            x = x.to(torch.int8)
            xabs = torch.abs(x)

            lower = torch.bitwise_and(
                input=xabs, other=torch.tensor(self.low_border, dtype=torch.int8)
            )

            upper = (
                torch.bitwise_and(
                    input=xabs, other=torch.tensor(self.high_border, dtype=torch.int8)
                )
                >> self.bits
            )

            lower = torch.copysign(lower, x)
            upper = torch.copysign(upper, x)

            x = torch.cat((upper, lower), 1)

        x = x / normalize_factor
        x = x.clamp(-1.0, 1.0 - 1.0 / normalize_factor)

        return x


class AdaptiveFixedPointNormalizer(nn.Module):
    "Simple feature normalizer for fixed point models"

    def __init__(self, normalize_bits: int = 8, num_features: int = 40) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=num_features, affine=True)
        self.normalize_bits = normalize_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        normalize_factor = 2.0 ** (self.normalize_bits - 1)
        x = self.bn(x)
        x = x / normalize_factor
        x = x.clamp(-1.0, 1.0 - 1.0 / normalize_factor)
        return x


class HistogramNormalizer(nn.Module):
    def __init__(self, bits: int = 8, bins: Optional[int] = None) -> None:
        super().__init__()

        self.bits = bits
        if bins is None:
            bins = min(2**bits, 2048)
        self.bins = bins

        self.register_buffer("histogram", torch.zeros(self.bins))
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.detach()

        new_min = torch.min(x)
        new_max = torch.max(x)

        self.min_val = torch.min(self.min_val, new_min)
        self.max_val = torch.max(self.max_val, new_max)

        histogram = torch.histc(
            x, min=int(self.min_val), max=int(self.max_val), bins=self.bins
        )

        self.histogram = self.histogram * 0.9 + histogram * 0.1
        print(self.min_val)
        print(self.max_val)

        return x_orig
