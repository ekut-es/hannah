import logging
from pytorch_lightning.callbacks import Callback
import distiller


class DistillerCallback(Callback):
    def __init__(self, config, fold_bn=-1):
        self.config = config

        self.fold_bn = fold_bn
        self.bn_frozen = False
        self.msglogger = logging.getLogger()

    def on_train_start(self, trainer, pl_module):
        pl_module.model.to(pl_module.device)

        if self.fold_bn >= 0 and not self.bn_frozen:
            self.bn_frozen = True
            self.msglogger.info("Applying batch norm folding")
            self.model = distiller.model_transforms.fold_batch_norms(
                pl_module.model,
                dummy_input=pl_module.example_input_array,
                inference=False,
            )
            self.msglogger.info("Folded model")
            self.msglogger.info(pl_module)

        self.msglogger.info("Activating compression scheduler")
        optimizers = trainer.optimizers
        if len(optimizers) != 1:
            raise Exception(
                "Distiller is only available when using  a single optimizers"
            )
        optimizer = optimizers[0]

        self.compression_scheduler = distiller.file_config(
            pl_module, optimizer, self.config
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

        if self.fold_bn == trainer.current_epoch and not self.bn_frozen:
            self.bn_frozen = True
            self.msglogger.info("Freezing batch norms")
            # save_model(log_dir, model, test_set, config=config, model_prefix="before_freeze_")

            def freeze_func(model):
                if isinstance(
                    model, distiller.quantization.sim_bn_fold.SimulatedFoldedBatchNorm
                ):
                    model.freeze()

            with torch.no_grad():
                self.model.apply(freeze_func)

            self.msglogger.info("Model after freezing")
            self.msglogger.info(pl_module)

    def on_epoch_end(self, trainer, pl_module):
        self.compression_scheduler.on_epoch_end(pl_module.current_epoch)
