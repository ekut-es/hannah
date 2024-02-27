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
from enum import Enum
from typing import Optional


class ManagementType(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


class CouplingType(Enum):
    DECOUPLED = "decoupled"
    COUPLED = "coupled"


class MemoryType:
    def __init__(
        self,
        size: int,
        scope: str,
        read_bw: int,
        write_bw: int,
        read_energy: int = 0,
        write_energy: int = 0,
        idle_energy: int = 0,
        latency: int = 1,
        area: int = 1,
        read_port=1,
        write_port=1,
        rw_port=0,
        management: ManagementType = ManagementType.EXPLICIT,
        coupling: CouplingType = CouplingType.DECOUPLED,
    ) -> None:
        self.size = size
        self.scope = scope
        self.management = management
        self.coupling = coupling
        self.size = size  # Size in bytes
        self.read_bw = read_bw
        self.write_bw = write_bw
        self.read_energy = read_energy
        self.write_energy = write_energy
        self.idle_energy = idle_energy
        self.latency = latency
        self.area = area
        self.read_port = read_port
        self.write_port = write_port
        self.rw_port = rw_port

        # TODO: add more attributes from the memory model
        # min_r_granularity, min_w_granularity is used for zigzag, for gpus we should model shared memory bank conflicts and global memory coalescing
