import logging
from pathlib import Path
from tempfile import TemporaryDirectory

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

    def run_batch(self, batch):
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
                result = self.run_batch(inputs=[batch[0]])
                target = pl_module.forward(batch[0])

                mse = torch.nn.functional.mse_loss(result, target, reduction="mean")
                for logger in pl_module.logger:
                    logger.log_metrics({"val_backend_mse": mse})

    def on_validation_epoch_end(self, trainer, pl_module):
        self.validation_epoch += 1

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx < self.test_batches:
            result = self.run_batch(inputs=[batch[0]])
            target = pl_module.forward(batch[0])

            mse = torch.nn.functional.mse_loss(result, target, reduction="mean")
            for logger in pl_module.logger:
                logger.log_metrics({"test_backend_mse": mse})


class TorchMobileBackend(InferenceBackendBase):
    """Inference backend for torch mobile"""

    def __init__(self, val_batches=1, test_batches=1, val_frequency=1):
        super().__init__(val_batches, test_batches, val_frequency)

        self.script_module = None

    def prepare(self, model):
        self.script_module = model.to_torchscript(method="trace")

    def run_batch(self, inputs=None):
        if inputs is None:
            return None
        return self.script_module(*inputs)


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
        self.use_acc_statistic_model  = use_acc_statistic_model
        self.use_acc_analytical_model = False
        self.use_acc_teda_data        = False
        self.performance = 1000000000.0
        self.power = 1000000000.0
        self.area = 1000000000.0
        self.accuracy = 0.0
        
    def _do_summary(self, pl_module, print=True):
        res = OrderedDict()

        if self.use_acc_statistic_model:
            self.performance = 1457.2   *self.cols**2 -33736.2  *self.cols -6.5      *self.bw_w**2 +65       *self.bw_w +170720.2
            self.power       = 1.469e-07*self.cols**2 +3.133e-06*self.cols +2.937e-07*self.bw_w**2 +2.175e-06*self.bw_w -1.514e-05
            self.area        = 792.9    *self.cols**2 +1026.6   *self.cols -122.5    *self.bw_w**2 +18941.7  *self.bw_w -63560.6
            self.accuracy    = 1.0
       # if use_acc_analytical_model:
       #     self.performance = Formula from the UltraTrail paper
        
        res["acc_performance"] = self.performance
        res["acc_power"]       = self.power
        res["acc_area"]        = self.area
        res["acc_accuracy"]    = self.accuracy
        return res
        
    def on_validation_epoch_end(self, trainer, pl_module):
        res = self._do_summary(pl_module, print=False)
        for k, v in res.items():
            pl_module.log(k, v)
        
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx < self.num_inferences:
            x = pl_module._extract_features(batch[0].cuda())
            x = pl_module.normalizer(x)
            y = pl_module.model(x)
            x = x.cpu()
            y = y.cpu()
            self.xs.append(x)
            self.ys.append(y)

    def on_test_end(self, trainer, pl_module):
        logging.info("Preparing ultratrail")
        # load backend package
        import sys

        sys.path.append(self.backend_dir)
        from backend.backend import UltraTrailBackend

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
            pl_module.model, pl_module.example_feature_array, verbose=True
        )
        backend.set_inputs_and_outputs(self.xs, self.ys)
        backend.prepare()
        backend.eda(self.standalone, self.rtl_simulation, self.synthesis, self.postsyn_simulation, self.power_estimation)
