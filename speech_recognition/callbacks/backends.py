import logging

from tempfile import TemporaryDirectory
from pathlib import Path

from pytorch_lightning import Callback
import torch.onnx
import torch.nn.functional as F
from onnx_tf.backend import prepare
import onnx
import tensorflow.lite as lite


class InferenceBackendBase(Callback):
    """ Base class to run val and test on a backend inference enginge """

    def __init__(
        self, limit_val_batches=0.01, limit_test_batches=0.01, val_frequency=10
    ):
        self.limit_val_batches = limit_val_batches
        self.limit_test_batches = limit_test_batches
        self.val_frequency = val_frequency
        self.validation_epoch = 0

    def run_batch(self, batch):
        raise NotImplementedError("run_batch is an abstract method")

    def prepare(self, module):
        raise NotImplementedError("prepare is an abstract method")

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.validation_epoch % self.val_frequency == 0:
            self.prepare(pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.validation_epoch % self.val_frequency == 0:
            result = self.run_batch(inputs=[batch[0]])
            result = torch.from_numpy(result[0])
            target = pl_module.forward(batch[0])

            mse = F.mse_loss(result, target, reduction="mean")
            print(mse)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.validation_epoch += 1

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        result = self.run_batch(inputs=[batch[0]])


class OnnxTFBackend(InferenceBackendBase):
    """Inference Backend for tensorflow"""

    def __init__(
        self,
        limit_val_batches=0.01,
        limit_test_batches=0.01,
        val_frequency=10,
        use_tf_lite=True,
    ):
        super(OnnxTFBackend, self).__init__(
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            val_frequency=10,
        )

        self.use_tf_lite = use_tf_lite
        self.tf_model = None
        self.interpreter = None

    def prepare(self, model):
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            logging.info("transfering model to onnx")
            dummy_input = model.example_input_array
            torch.onnx.export(model, dummy_input, tmp_dir / "model.onnx", verbose=True)
            logging.info("Creating tf-protobuf")
            onnx_model = onnx.load(tmp_dir / "model.onnx")
            self.tf_model = prepare(onnx_model)

    def run_batch(self, inputs=[]):
        logging.info("running tf backend on batch")

        result = self.tf_model.run(inputs=inputs)

        return result
