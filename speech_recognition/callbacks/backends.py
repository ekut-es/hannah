import math
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from speech_recognition.callbacks.summaries import walk_model

from collections import OrderedDict

import torch.onnx
from pytorch_lightning import Callback

try:
    import onnx
except ModuleNotFoundError:
    onnx = None

try:
    import onnx_tf.backend as tf_backend
except ModuleNotFoundError:
    tf_backend = None

try:
    import onnxruntime.backend as onnxrt_backend
except ModuleNotFoundError:
    onnxrt_backend = None

from ..models.factory.qat import QAT_MODULE_MAPPINGS


def symbolic_batch_dim(model):
    sym_batch_dim = "N"

    inputs = model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_param = sym_batch_dim


class InferenceBackendBase(Callback):
    """ Base class to run val and test on a backend inference engine """

    def __init__(self, val_batches=1, test_batches=1, val_frequency=10):
        self.test_batches = test_batches
        self.val_batches = val_batches
        self.val_frequency = val_frequency
        self.validation_epoch = 0

    def run_batch(self, inputs=None):
        raise NotImplementedError("run_batch is an abstract method")

    def prepare(self, module):
        raise NotImplementedError("prepare is an abstract method")

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.val_batches > 0:
            if self.validation_epoch % self.val_frequency == 0:
                self.prepare(pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx < self.val_batches:
            if self.validation_epoch % self.val_frequency == 0:
                result = self.run_batch(inputs=batch[0])
                target = pl_module.forward(batch[0].to(pl_module.device))

                mse = torch.nn.functional.mse_loss(result, target, reduction="mean")
                pl_module.log("val_backend_mse", mse)
                logging.info("val_backend_mse: %f", mse)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.validation_epoch += 1

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx < self.test_batches:
            result = self.run_batch(inputs=batch[0])
            target = pl_module(batch[0].to(pl_module.device))

            mse = torch.nn.functional.mse_loss(result, target, reduction="mean")
            pl_module.log("test_backend_mse", mse)
            logging.info("test_backend_mse: %f", mse)


class TorchMobileBackend(InferenceBackendBase):
    """Inference backend for torch mobile"""

    def __init__(self, val_batches=1, test_batches=1, val_frequency=1):
        super().__init__(val_batches, test_batches, val_frequency)

        self.script_module = None

    def prepare(self, model):
        logging.info("Preparing model for target")
        self.script_module = model.to_torchscript(method="trace")

    def run_batch(self, inputs=None):
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
        logging.info("running onnxruntime backend on batch")

        result = self.onnxrt_model.run(inputs=[input.numpy() for input in inputs])
        result = [torch.from_numpy(res) for res in result]
        return result


class TRaxUltraTrailBackend(Callback):
    """TRax UltraTrail backend"""

    def __init__(
        self,
        backend_dir,
        teda_dir,
        standalone,
        rtl_simulation,
        synthesis,
        postsyn_simulation,
        power_estimation,
        num_inferences,
        bw_w,
        bw_b,
        bw_f,
        cols,
        rows,
        period,
        macro_type,
        use_acc_statistic_model,
        use_acc_analytical_model,
    ):
        self.backend_dir = backend_dir
        self.teda_dir = Path(teda_dir)
        self.standalone = standalone
        self.rtl_simulation = rtl_simulation
        self.synthesis = synthesis
        self.postsyn_simulation = postsyn_simulation
        self.power_estimation = power_estimation
        self.num_inferences = num_inferences
        self.bw_w = bw_w
        self.bw_b = bw_b
        self.bw_f = bw_f
        self.cols = cols
        self.rows = rows
        self.period = period
        self.macro_type = macro_type
        self.xs = []
        self.ys = []
        self.use_acc_statistic_model = use_acc_statistic_model
        self.use_acc_analytical_model = use_acc_analytical_model
        self.use_acc_teda_data = False
        # Performance in clock cycles
        # Power in W
        # Area in µm^2
        self.clock_cycles = 1000000000.0
        self.power = 1000000000.0
        self.area = 1000000000.0
        self.accuracy = 0.0

    def get_analytical_clock_cycles(self, pl_module):
        model = pl_module.model
        dummy_input = pl_module.example_feature_array
        df = walk_model(pl_module.model, dummy_input)

        clock_cycles = 0
        for name, layer in pl_module.model.named_modules():
            if isinstance(layer, torch.nn.Conv1d) or isinstance(layer, torch.nn.Linear):
                if isinstance(layer, torch.nn.Linear):
                    C = layer.in_features
                    K = layer.out_features
                    F = 1
                    C_w = 1
                    s = 1
                    pad = 0
                else:
                    df_index = df[df["Name"] == name].index[0]
                    layer_ifm = df["IFM"][df_index]
                    C_w = layer_ifm[2]
                    C = layer.in_channels
                    K = layer.out_channels
                    F = layer.kernel_size[0]
                    s = layer.stride[0]
                    pad = 1 if layer.padding[0] > 0 else 0

                C_w_mod = C_w + pad * 2 * (F // 2)
                a_w = math.floor(((C_w_mod - F) // s) + 1)
                C_w_b = F // 2
                a_p_b = pad * math.floor(((C_w_b - 1) / s) + 1)
                MAC_not_b = 0
                for i in range(a_p_b):
                    MAC_not_b += (F // 2) - s * i

                F_w = a_w * s + F - s
                C_w_e = F_w - C_w - C_w_b
                a_p_e = pad * math.floor(((C_w_e - 1) / s) + 1)
                MAC_not_e = 0
                for i in range(a_p_e):
                    MAC_not_e += (F // 2) - s * i - (C_w_b - C_w_e)

                t_l = 1 + math.ceil(C / self.rows) * math.ceil(K / self.cols) * (
                    a_w * F - MAC_not_b - MAC_not_e
                )
                clock_cycles += t_l
        return clock_cycles

    def _do_summary(self, pl_module):
        res = OrderedDict()
        if self.use_acc_statistic_model:
            self.clock_cycles = (
                1457.2 * self.cols ** 2
                - 33736.2 * self.cols
                - 6.5 * self.bw_w ** 2
                + 65 * self.bw_w
                + 170720.2
            )
            self.power = (
                1.469e-07 * self.cols ** 2
                + 3.133e-06 * self.cols
                + 2.937e-07 * self.bw_w ** 2
                + 2.175e-06 * self.bw_w
                - 1.514e-05
            )
            self.area = (
                792.9 * self.cols ** 2
                + 1026.6 * self.cols
                - 122.5 * self.bw_w ** 2
                + 18941.7 * self.bw_w
                - 63560.6
            )
            self.accuracy = 1.0

        # Assumption: Analytical model is more precise than statistical
        if self.use_acc_analytical_model:
            self.clock_cycles = self.get_analytical_clock_cycles(pl_module)

        # Wait for movement of the whole code to the backend
        # if self.use_acc_teda_data:

        res["acc_clock_cycles"] = self.clock_cycles
        res["acc_power"] = self.power
        res["acc_area"] = self.area
        res["acc_accuracy"] = self.accuracy
        return res

    def on_validation_epoch_end(self, trainer, pl_module):
        res = self._do_summary(pl_module)
        for k, v in res.items():
            pl_module.log(k, v)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if len(self.xs) < self.num_inferences:
            x = pl_module._extract_features(batch[0])
            x = pl_module.normalizer(x)
            y = pl_module.model(x)

            x = x.cpu().split(1)
            y = y.cpu().split(1)
            y = [t.squeeze() for t in y]

            self.xs.extend(x)
            self.ys.extend(y)

    def on_test_end(self, trainer, pl_module):
        logging.info("Preparing ultratrail")
        # load backend package
        import sys

        sys.path.append(self.backend_dir)
        from backend.backend import UltraTrailBackend

        model = pl_module.model
        if hasattr(model, "qconfig"):
            # Removing qconfig produces a normal FloatModule
            model = torch.quantization.convert(
                model, mapping=QAT_MODULE_MAPPINGS, remove_qconfig=True
            )

        # execute backend
        backend = UltraTrailBackend(
            teda=self.teda_dir,
            bw_w=self.bw_w,
            bw_b=self.bw_b,
            bw_f=self.bw_f,
            cols=self.cols,
            rows=self.rows,
            period=self.period,
            macro_type=self.macro_type,
        )
        backend.set_model(
            model.cpu(), pl_module.example_feature_array.cpu(), verbose=True
        )
        backend.set_inputs_and_outputs(self.xs, self.ys)
        backend.prepare()
        backend.eda(
            self.standalone,
            self.rtl_simulation,
            self.synthesis,
            self.postsyn_simulation,
            self.power_estimation,
        )
