from pytorch_lightning.callbacks import Callback
import distiller


class DistillerCallback(Callback):
    # def __init__(self, lit_module):
    #     super().__init__()
    #     self.lit_module = lit_module
    #     self.hparams = lit_module.hparams
    #     self.model = lit_module.model
    #     self.optimizer = lit_module.optimizer
    #     self.msglogger = lit_module.msglogger

    def on_train_start(self, trainer, pl_module):
        pl_module.model.to(pl_module.device)
        pl_module.msglogger.info("Activating compression scheduler")
        pl_module.compression_scheduler = distiller.file_config(
            pl_module.model, pl_module.optimizer, pl_module.hparams["compress"]
        )

    def on_epoch_start(self, trainer, pl_module):
        pl_module.compression_scheduler.on_epoch_begin(trainer.current_epoch)

    def on_batch_start(self, trainer, pl_module):
        pl_module.compression_scheduler.on_minibatch_begin(
            trainer.current_epoch,
            trainer.batch_idx,
            pl_module.batches_per_epoch,
        )

    def on_batch_end(self, trainer, pl_module):
        pl_module.compression_scheduler.on_minibatch_end(
            trainer.current_epoch,
            trainer.batch_idx,
            pl_module.batches_per_epoch,
        )

    def on_epoch_end(self, trainer, pl_module):
        pl_module.compression_scheduler.on_epoch_end(pl_module.current_epoch)
