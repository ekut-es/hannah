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

from .base_sampler import Sampler, SearchResult


class RandomSampler(Sampler):
    def __init__(
        self,
        parent_config,
        search_space,
        parametrization,
        output_folder=".",
    ) -> None:
        super().__init__(parent_config=parent_config, search_space=search_space, output_folder=output_folder)
        self.parametrization = parametrization

        if (self.output_folder / "history.yml").exists():
            self.load()

    def next_parameters(self):
        parameter_values = {}
        for key, param in self.parametrization.items():
            parameter_values[key] = param.sample()
        return parameter_values, []
