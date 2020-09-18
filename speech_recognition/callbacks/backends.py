from pytorch_lightning import Callback
import logging


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

    def on_validation_batch_end(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        result = self.run_batch(batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.validation_epoch += 1

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        result = self.run_batch(batch)


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

    def run_batch(self, batch):
        print("running tf backend on batch")
        print(batch)
