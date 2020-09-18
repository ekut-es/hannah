import logging

from pytorch_lightning import Callback
import torch.onnx
from onnx_tf.backend import prepare
import onnx


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
        self.prepare(pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        result = self.run_batch(inputs=[batch[0]], outputs=[batch[2]])

    def on_validation_epoch_end(self, trainer, pl_module):
        self.validation_epoch += 1

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        result = self.run_batch(inputs=[batch[0]], outputs=[batch[2]])


class OnnxTFBackend(InferenceBackendBase):
    """Inference Backend for tensorflow"""

    def __init__(
        self, limit_val_batches=0.01, limit_test_batches=0.01, val_frequency=10
    ):
        super(OnnxTFBackend, self).__init__(
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            val_frequency=10,
        )

        self.tf_model = None

    def prepare(self, model):
        print("transfering model to onnx")
        dummy_input = model.example_input_array
        torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)
        print("Creating tf-protobuf")
        onnx_model = onnx.load("model.onnx")
        self.tf_model = prepare(onnx_model)

    def run_batch(self, inputs=[], outputs=[]):
        print("running tf backend on batch")

        result = self.tf_model.run(inputs=inputs)
        print(result)
