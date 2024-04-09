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


class InferenceBackendBase(AbstractBackend, Callback):
    """Base class to wrap backends as a lightning callback"""

    def __init__(
        self, val_batches=1, test_batches=1, val_frequency=10, tune: bool = True
    ):
        self.test_batches = test_batches
        self.val_batches = val_batches
        self.val_frequency = val_frequency
        self.validation_epoch = 0
        self.tune = tune

    def on_validation_epoch_start(self, trainer, pl_module):
        if not self.tune:
            return

        if self.val_batches > 0:
            if self.validation_epoch % self.val_frequency == 0:
                pl_module = self.quantize(pl_module)
                self.prepare(pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=-1
    ):
        if not self.tune:
            return

        if batch_idx < self.val_batches:
            if self.validation_epoch % self.val_frequency == 0:
                result = self.run_batch(inputs=batch[0])
                if not isinstance(result, torch.Tensor):
                    logging.warning("Could not calculate MSE on target device")
                    return
                target = pl_module.forward(batch[0].to(pl_module.device))
                mse = torch.nn.functional.mse_loss(
                    result.to(pl_module.device),
                    target.to(pl_module.device),
                    reduction="mean",
                )
                pl_module.log("val_backend_mse", mse)
                logging.info("val_backend_mse: %f", mse)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.validation_epoch += 1

    def on_test_epoch_start(self, trainer, pl_module):
        logger.info("Exporting module")

        pl_module = self.quantize(pl_module)
        self.prepare(pl_module)
        self.export()

    def quantize(self, pl_module: torch.nn.Module) -> torch.nn.Module:
        qconfig_mapping = getattr(pl_module, "qconfig_mapping", None)
        if qconfig_mapping is None:
            logger.info("No qconfig found in module, leaving module unquantized")
            return pl_module

        pl_module = copy.deepcopy(pl_module)
        pl_module.cpu()

        logger.info("Quantizing module")

        example_inputs = next(iter(pl_module.train_dataloader()))[0]

        model = torch.ao.quantization.quantize_fx.prepare_fx(
            pl_module.model, qconfig_mapping, example_inputs
        )
        model = torch.ao.quantization.quantize_fx.convert_fx(model)
        pl_module.model = model

        return pl_module

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=-1
    ):
        if batch_idx < self.test_batches:
            # decode batches from target device
            if isinstance(batch, Mapping) or isinstance(batch, dict):
                inputs = batch["data"]
            else:
                inputs = batch[0]

            result = self.run_batch(inputs=inputs)
            target = pl_module(inputs.to(pl_module.device))
            target = target[: result.shape[0]]

            mse = torch.nn.functional.mse_loss(
                result.to(pl_module.device),
                target.to(pl_module.device),
                reduction="mean",
            )
            pl_module.log("test_backend_mse", mse)
            logging.info("test_backend_mse: %f", mse)
