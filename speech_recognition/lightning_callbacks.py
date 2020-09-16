from pytorch_lightning.callbacks import Callback
import distiller


class DistillerCallback(Callback):
    def __init__(self, lit_module):
        super().__init__()
        self.lit_module = lit_module
        self.hparams = lit_module.hparams
        self.model = lit_module.model
        self.optimizer = lit_module.optimizer
        self.msglogger = lit_module.msglogger

    def on_train_start(self, trainer, pl_module):
        self.model.to(self.lit_module.device)
        self.msglogger.info("Activating compression scheduler")
        self.lit_module.compression_scheduler = distiller.file_config(
            self.model, self.optimizer, self.hparams["compress"]
        )

    def on_epoch_start(self, trainer, pl_module):
        self.lit_module.compression_scheduler.on_epoch_begin(
            self.lit_module.current_epoch
        )

    def on_batch_end(self, trainer, pl_module):
        self.lit_module.compression_scheduler.on_minibatch_end(
            self.lit_module.current_epoch,
            self.lit_module.batch_idx,
            self.lit_module.batches_per_epoch,
        )

    def on_epoch_end(self, trainer, pl_module):
        self.lit_module.compression_scheduler.on_epoch_end(
            self.lit_module.current_epoch
        )
