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
from hydra.utils import instantiate


def profile_backend(config, lit_module):
    metrics = {}
    if config.get("backend"):
        backend = instantiate(config.backend)
        backend.prepare(lit_module)

        backend_results = backend.profile(lit_module.example_input_array)  # noqa

        metrics = backend_results.metrics

    return metrics