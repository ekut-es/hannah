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
import copy
import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Union

import torch
from lightning.pytorch import Callback

from ..modules.base import ClassifierModule

_PATH = Union[pathlib.Path, str]


logger = logging.getLogger(__name__)


class ProfilingResult(NamedTuple):
    """Result of a profiling run

    Attributes:
        outputs: the outputs of the model on the given input batch
        metrics: a dictionary containing the combined metrics obtained from the profiling run
        profile: the raw profile in a backend-specific format
    """

    outputs: Union[torch.tensor, Sequence[torch.tensor]]
    metrics: Mapping[str, float]
    profile: Optional[Any]


class AbstractBackend(ABC):
    @abstractmethod
    def prepare(self, module: ClassifierModule):
        """
        Prepare the model for execution on the target device

        Args:
          module: the classifier module to be exported

        """
        ...

    @abstractmethod
    def run(self, *inputs) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """
        Run a batch on the target device

        Args:
          inputs: a list of torch tensors representing the inputs to be run on the target device, each tensor represents a whole batched input, so for models taking 1 parameter, the list will contain 1 tensor of shape (batch_size, *input_shape)

        Returns: the output(s) of the model as a torch tensor or a Sequence of torch tensors for models producing multiple outputs

        """
        ...

    @abstractmethod
    def profile(self, *inputs: torch.Tensor) -> ProfilingResult:
        """Do a profiling run on the target device

        Args:
            inputs: a list of torch tensors representing the inputs to be run on the target device, each tensor represents a whole batched input, so for models taking 1 parameter, the list will contain 1 tensor of shape (batch_size, *input_shape)

        Returns: a ProfilingResult object containing the outputs of the model, the metrics obtained from the profiling run and the raw profile in a backend-specific format
        """
        ...

    @classmethod
    @abstractmethod
    def available(cls) -> bool:
        """
        Check if the backend is available

        Returns: True if the backend is available, False otherwise

        """
        ...

    def export(self) -> None:
        """
        Export the model through the target backend
        """

        logger.critical("Exporting model is not implemented for this backend")


class InferenceBackendBase(AbstractBackend):
    """Base class for backends, it is only here for backwards compatibility reasons, use AbstractBackend instead"""

    pass
