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
from typing import Optional, Union

import numpy as np

from hannah.nas.functional_operators.lazy import lazy
from hannah.nas.parameters.parameters import Parameter


def channels_per_group(n):
    channels = []
    i = n
    while i > 0:
        x = n % i
        if x == 0:
            channels.append(int(n / i))
        i -= 1
    return channels


class Groups(Parameter):
    def __init__(
        self,
        in_channels: Union[int, Parameter],
        out_channels: Union[int, Parameter],
        name: Optional[str] = "",
        rng: Optional[Union[np.random.Generator, int]] = None,
    ):
        super().__init__(name, rng)
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.current_value = self.sample()

    def get_possible_values(self):
        in_channels = lazy(self.in_channels)
        out_channels = lazy(self.out_channels)
        possible_values_in = channels_per_group(in_channels)
        possible_values_out = channels_per_group(out_channels)
        possible_values = list(
            set(possible_values_in).intersection(possible_values_out)
        )
        return possible_values

    def instantiate(self):
        possible_values = self.get_possible_values()
        if self.current_value not in possible_values:
            diff = np.abs(np.array(possible_values)) - self.current_value
            self.current_value = int(possible_values[np.argmin(diff)])
        return self.current_value

    def sample(self):
        possible_values = self.get_possible_values()
        self.current_value = int(np.random.choice(possible_values))
        return self.current_value

    def check(self, value):
        possible_values = self.get_possible_values()
        if value not in possible_values:
            raise ValueError(
                "{} channels per group not valid with {} in channels and {} out channels".format(
                    value, self.in_channels.evaluate(), self.out_channels.evaluate()
                )
            )

    def set_current(self, x):
        possible_values = self.get_possible_values()
        if x not in possible_values:
            diff = np.abs(np.array(possible_values)) - x
            self.current_value = int(possible_values[np.argmin(diff)])
        else:
            self.current_value = int(x)

    def from_float(self, val):
        possible_values = self.get_possible_values()
        val = int(val * (len(possible_values) - 1))

        return possible_values[val]
