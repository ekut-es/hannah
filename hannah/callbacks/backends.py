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

import copy
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.onnx
from pytorch_lightning import Callback

try:
    import onnx  # pytype: disable=import-error
except ModuleNotFoundError:
    onnx = None

try:
    import onnx_tf.backend as tf_backend  # pytype: disable=import-error
except ModuleNotFoundError:
    tf_backend = None

try:
    import onnxruntime.backend as onnxrt_backend  # pytype: disable=import-error
except ModuleNotFoundError:
    onnxrt_backend = None

from ..models.factory.qat import QAT_MODULE_MAPPINGS

logger = logging.getLogger(__name__)


def symbolic_batch_dim(model):
    """

    Args:
      model:

    Returns:

    """
    sym_batch_dim = "N"

    inputs = model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_param = sym_batch_dim


class InferenceBackendBase(Callback):
    """Base class to run val and test on a backend inference engine"""

    def __init__(
        self, val_batches=1, test_batches=1, val_frequency=10, tune: bool = True
    ):
        self.test_batches = test_batches
        self.val_batches = val_batches
        self.val_frequency = val_frequency
        self.validation_epoch = 0
        self.tune = tune

    def run_batch(self, inputs=None):
        """

        Args:
          inputs:  (Default value = None)

        Returns:

        """
        raise NotImplementedError("run_batch is an abstract method")

    def prepare(self, module):
        """

        Args:
          module:

        Returns:

        """
        raise NotImplementedError("prepare is an abstract method")

    def on_validation_epoch_start(self, trainer, pl_module):
        """

        Args:
          trainer:
          pl_module:

        Returns:

        """
        if not self.tune:
            return

        if self.val_batches > 0:
            if self.validation_epoch % self.val_frequency == 0:
                pl_module = self.quantize(pl_module)
                self.prepare(pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """

        Args:
          trainer:
          pl_module:
          outputs:
          batch:
          batch_idx:
          dataloader_idx:

        Returns:

        """
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
        """

        Args:
          trainer:
          pl_module:

        Returns:

        """
        self.validation_epoch += 1

    def on_test_epoch_start(self, trainer, pl_module):
        """

        Args:
          trainer:
          pl_module:

        Returns:

        """
        pl_module = self.quantize(pl_module)
        self.prepare(pl_module)
        self.export()

    def quantize(self, pl_module: torch.nn.Module) -> torch.nn.Module:
        """

        Args:
          pl_module: torch.nn.Module to quantize

        Returns: quantized  torch.nn.Module

        """
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
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """

        Args:
          trainer:
          pl_module:
          outputs:
          batch:
          batch_idx:
          dataloader_idx:

        Returns:

        """
        if batch_idx < self.test_batches:
            result = self.run_batch(inputs=batch[0])
            target = pl_module(batch[0].to(pl_module.device))
            target = target[: result.shape[0]]

            mse = torch.nn.functional.mse_loss(
                result.to(pl_module.device),
                target.to(pl_module.device),
                reduction="mean",
            )
            pl_module.log("test_backend_mse", mse)
            logging.info("test_backend_mse: %f", mse)

    def export(self) -> None:
        """
        Export the model through the target backend
        """
        logger.critical("Exporting model is not implemented for this backend")


class TorchMobileBackend(InferenceBackendBase):
    """Inference backend for torch mobile"""

    def __init__(self, val_batches=1, test_batches=1, val_frequency=1):
        super().__init__(val_batches, test_batches, val_frequency)

        self.script_module = None

    def prepare(self, model):
        """
        Args:
          model (torch.nn.Module): nn.Module to be exported

        Returns (None)
        """
        logging.info("Preparing model for target")
        self.script_module = model.to_torchscript(method="trace")

    def run_batch(self, inputs=None):
        """

        Args:
          inputs:  (Default value = None)

        Returns:

        """
        if inputs is None:
            logging.critical("Backend batch is empty")
            return None

        return self.script_module(inputs)


class OnnxTFBackend(InferenceBackendBase):
    """Inference Backend for tensorflow"""

    def __init__(self, val_batches=1, test_batches=1, val_frequency=10):
        super(OnnxTFBackend, self).__init__(
            val_batches=val_batches,
            test_batches=test_batches,
            val_frequency=val_frequency,
        )

        self.tf_model = None
        self.interpreter = None

        if onnx is None or tf_backend is None:
            raise Exception(
                "Could not find required libraries for onnx-tf backend please install with poetry instell -E tf-backend"
            )

    def prepare(self, model):
        """

        Args:
          model:

        Returns:

        """
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            logging.info("transfering model to onnx")
            dummy_input = model.example_input_array
            torch.onnx.export(model, dummy_input, tmp_dir / "model.onnx", verbose=False)
            logging.info("Creating tf-protobuf")
            onnx_model = onnx.load(tmp_dir / "model.onnx")
            symbolic_batch_dim(onnx_model)
            self.tf_model = tf_backend.prepare(onnx_model)

    def run_batch(self, inputs):
        """

        Args:
          inputs:

        Returns:

        """
        logging.info("running tf backend on batch")

        result = self.tf_model.run(inputs=inputs)
        result = [torch.from_numpy(res) for res in result]
        return result


class OnnxruntimeBackend(InferenceBackendBase):
    """Inference Backend for tensorflow"""

    def __init__(
        self, val_batches=1, test_batches=1, val_frequency=10, use_tf_lite=True
    ):
        super(OnnxruntimeBackend, self).__init__(
            val_batches=val_batches, test_batches=test_batches, val_frequency=10
        )

        self.onnxrt_model = None

        if onnx is None or onnxrt_backend is None:
            raise Exception(
                "Could not find required libraries for onnxruntime backend please install with poetry instell -E onnxrt-backend"
            )

    def prepare(self, model):
        """

        Args:
          model:

        Returns:

        """
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            logging.info("transfering model to onnx")
            dummy_input = model.example_input_array
            torch.onnx.export(model, dummy_input, tmp_dir / "model.onnx", verbose=False)
            logging.info("Creating onnxrt-model")
            onnx_model = onnx.load(tmp_dir / "model.onnx")
            symbolic_batch_dim(onnx_model)
            self.onnxrt_model = onnxrt_backend.prepare(onnx_model)

    def run_batch(self, inputs=None):
        """

        Args:
          inputs:  (Default value = None)

        Returns

        """
        logging.info("running onnxruntime backend on batch")

        result = self.onnxrt_model.run(inputs=[input.numpy() for input in inputs])
        result = [torch.from_numpy(res) for res in result]
        return result
