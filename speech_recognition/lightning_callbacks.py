from pytorch_lightning.callbacks import Callback
import distiller


class DistillerCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        pl_module.model.to(pl_module.device)
        pl_module.msglogger.info("Activating compression scheduler")
        self.compression_scheduler = distiller.file_config(
            pl_module.model, pl_module.optimizer, pl_module.hparams["compress"]
        )

    def on_epoch_start(self, trainer, pl_module):
        self.compression_scheduler.on_epoch_begin(trainer.current_epoch)

    def on_batch_start(self, trainer, pl_module):
        self.compression_scheduler.on_minibatch_begin(
            trainer.current_epoch,
            trainer.batch_idx,
            pl_module.batches_per_epoch,
        )

    def on_before_backward(self, trainer, pl_module, loss):
        self.compression_scheduler.before_backward_pass(
            trainer.current_epoch, trainer.batch_idx, pl_module.batches_per_epoch, loss
        )

    def on_batch_end(self, trainer, pl_module):
        self.compression_scheduler.on_minibatch_end(
            trainer.current_epoch,
            trainer.batch_idx,
            pl_module.batches_per_epoch,
        )

    def on_epoch_end(self, trainer, pl_module):
        self.compression_scheduler.on_epoch_end(pl_module.current_epoch)
