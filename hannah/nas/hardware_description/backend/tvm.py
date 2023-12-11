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
import logging

from ..device import Device

logger = logging.getLogger(__name__)


class BackendGenerator:
    def __init__(self):
        pass

    def rewrite(self, device: Device):
        pass


class TVMBackendGenerator:
    pass


def build_tvm_patterns(device: Device):
    generator = TVMBackendGenerator()

    pattern_table = generator.generate(device)

    return pattern_table
