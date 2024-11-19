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
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from hannah.nas.hardware_description.device import Device


class DescriptionBackend(ABC):
    """Abstract base class for generating tool specific descriptions from target devices."""

    @abstractmethod
    def generate(self, device: Device) -> Any:
        """Generates a tool specific description from a target device meta model."""
        pass

    def save(self, device: Device, path: str) -> None:
        """Saves a tool specific description to a file. if supported by the backend."""

        raise NotImplementedError(f"Saving is not supported by {self}.")

    def __str__(self) -> str:
        return self.__class__.__name__
