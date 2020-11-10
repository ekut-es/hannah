import logging
from pytorch_lightning.callbacks import Callback
import torch
import distiller
from omegaconf import DictConfig, OmegaConf


class DistillerCallback(Callback):
    def __init__(self, config: DictConfig):
        self.config = config
    
        # torch.autograd.set_detect_anomaly(True)

        self.fold_bn = config.get("fold_bn", None)
        self.bn_frozen = False
        self.msglogger = logging.getLogger()

    def on_train_start(self, trainer, pl_module):
        pl_module.model.to(pl_module.device)

        train_set_len = len(pl_module.train_dataloader()) 

        if isinstance(self.fold_bn, float):
            self.msglogger.info("Applying batch norm folding")
            self.model = distiller.model_transforms.fold_batch_norms(
                pl_module.model,
                dummy_input=pl_module.example_feature_array,
                inference=False,
                freeze_bn_delay = trainer.max_epochs * self.fold_bn * train_set_len
            )
            self.msglogger.info("Folded model")
            self.msglogger.info(pl_module)

        self.msglogger.info("Activating compression scheduler")
        optimizers = trainer.optimizers
        if len(optimizers) != 1:
            raise Exception(
                "Distiller is only available when using  a single optimizer"
            )
        optimizer = optimizers[0]

        self.compression_scheduler = distiller.dict_config(
            pl_module, optimizer, OmegaConf.to_container(self.config)
        )
        self.msglogger.info("Compressed Model")
        self.msglogger.info(pl_module)


    def on_epoch_start(self, trainer, pl_module):
        self.compression_scheduler.on_epoch_begin(trainer.current_epoch)

    def on_batch_start(self, trainer, pl_module):
        self.compression_scheduler.on_minibatch_begin(
            trainer.current_epoch, trainer.batch_idx, pl_module.batches_per_epoch
        )

    def on_before_backward(self, trainer, pl_module, loss):
        self.compression_scheduler.before_backward_pass(
            trainer.current_epoch, trainer.batch_idx, pl_module.batches_per_epoch, loss
        )

    def on_batch_end(self, trainer, pl_module):
        self.compression_scheduler.on_minibatch_end(
            trainer.current_epoch, trainer.batch_idx, pl_module.batches_per_epoch
        )

    def on_epoch_end(self, trainer, pl_module):
        self.compression_scheduler.on_epoch_end(pl_module.current_epoch)













