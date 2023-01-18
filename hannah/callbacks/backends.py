#
# Copyright (c) 2023 Hannah contributors.
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
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

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

    def __init__(self, val_batches=1, test_batches=1, val_frequency=10):
        self.test_batches = test_batches
        self.val_batches = val_batches
        self.val_frequency = val_frequency
        self.validation_epoch = 0

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
        if self.val_batches > 0:
            if self.validation_epoch % self.val_frequency == 0:
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
        if batch_idx < self.val_batches:
            if self.validation_epoch % self.val_frequency == 0:
                result = self.run_batch(inputs=batch[0])
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
        self.prepare(pl_module)

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

            mse = torch.nn.functional.mse_loss(
                result.to(pl_module.device),
                target.to(pl_module.device),
                reduction="mean",
            )
            pl_module.log("test_backend_mse", mse)
            logging.info("test_backend_mse: %f", mse)


class TorchMobileBackend(InferenceBackendBase):
    """Inference backend for torch mobile"""

    def __init__(self, val_batches=1, test_batches=1, val_frequency=1):
        super().__init__(val_batches, test_batches, val_frequency)

        self.script_module = None

    def prepare(self, model):
        """

        Args:
          model:

        Returns:

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

        Returns:

        """
        logging.info("running onnxruntime backend on batch")

        result = self.onnxrt_model.run(inputs=[input.numpy() for input in inputs])
        result = [torch.from_numpy(res) for res in result]
        return result


class TRaxUltraTrailBackend(Callback):
    """TRax UltraTrail backend"""

    def __init__(
        self,
        backend_dir,
        standalone,
        rtl_simulation,
        synthesis,
        postsyn_simulation,
        power_estimation,
        num_inferences,
        cols,
        rows,
        period,
        macro_type,
        use_acc_statistic_model,
        use_acc_analytical_model,
        use_acc_teda_data,
    ):
        self.backend_dir = backend_dir
        self.standalone = standalone
        self.rtl_simulation = rtl_simulation
        self.synthesis = synthesis
        self.postsyn_simulation = postsyn_simulation
        self.power_estimation = power_estimation
        self.num_inferences = num_inferences
        self.bw_w = None  # These are exectracted from models qconfig
        self.bw_b = None
        self.bw_f = None
        self.cols = cols
        self.rows = rows if rows is not None else cols
        self.period = period
        self.macro_type = macro_type
        self.xs = []
        self.ys = []
        self.use_acc_statistic_model = use_acc_statistic_model
        self.use_acc_analytical_model = use_acc_analytical_model
        self.use_acc_teda_data = use_acc_teda_data
        self.enable_file_generation = True

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
        if len(self.xs) < self.num_inferences:
            x = pl_module._extract_features(batch[0].to(pl_module.device))
            x = pl_module.normalizer(x)
            y = pl_module.model(x)

            x = x.cpu().split(1)
            y = y.cpu().split(1)
            y = [t.squeeze() for t in y]

            self.xs.extend(x)
            self.ys.extend(y)

    def _run(self, pl_module):
        """

        Args:
          pl_module:

        Returns:

        """
        # load backend package
        sys.path.append(self.backend_dir)
        from backend.backend import UltraTrailBackend  # pytype: disable=import-error

        classes = pl_module.num_classes
        model = pl_module.model
        mac_mode = "FIXED_POINT"
        if hasattr(model, "qconfig"):
            # Set UltraTrail mac and bit configuration depending on qconfig
            mac_mode = (
                "POWER_OF_TWO"
                if model.qconfig.weight.p.keywords["power_of_2"]
                else "FIXED_POINT"
            )
            self.bw_w = model.qconfig.weight.p.keywords["bits"]
            self.bw_b = model.qconfig.bias.p.keywords["bits"]
            self.bw_f = model.qconfig.activation.p.keywords["bits"]

            # Removing qconfig produces a normal FloatModule
            model = torch.quantization.convert(
                model, mapping=QAT_MODULE_MAPPINGS, remove_qconfig=True
            )

        if mac_mode == "POWER_OF_TWO":
            logging.critical(
                "PO2 quantization is enabled. Check that quantization range matches bw_wide_q"
            )

        # execute backend
        backend = UltraTrailBackend(
            bw_w=self.bw_w,
            bw_b=self.bw_b,
            bw_f=self.bw_f,
            cols=self.cols,
            rows=self.rows,
            period=self.period,
            mac_mode=mac_mode,
            macro_type=self.macro_type,
            classes=classes,
            enable_file_generation=self.enable_file_generation,
        )

        backend.set_model(
            model.cpu(), pl_module.example_feature_array.cpu(), verbose=True
        )
        backend.set_inputs_and_outputs(self.xs, self.ys)
        backend.prepare()
        if (
            self.use_acc_teda_data
            or self.rtl_simulation
            or self.synthesis
            or self.postsyn_simulation
            or self.power_estimation
        ):
            backend.eda(
                self.standalone,
                self.rtl_simulation,
                self.synthesis,
                self.postsyn_simulation,
                self.power_estimation,
            )

        res = backend._do_summary(
            self.use_acc_statistic_model,
            self.use_acc_analytical_model,
            self.use_acc_teda_data,
            self.rtl_simulation,
            self.synthesis,
            self.power_estimation,
        )
        return res

    def estimate(self, pl_module):
        """

        Args:
          pl_module:

        Returns:

        """
        input = pl_module.example_feature_array
        pl_module.eval()
        output = pl_module.model(input)
        self.xs.append(input)
        self.ys.append(output.squeeze())
        self.enable_file_generation = False
        return self._run(pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        """

        Args:
          trainer:
          pl_module:

        Returns:

        """
        logging.info("Preparing ultratrail")
        res = self._run(pl_module)

        logging.info("Ultratrail metrics")
        for k, v in res.items():
            pl_module.log(k, float(v))
            logging.info("%s: %s", str(k), str(v))
