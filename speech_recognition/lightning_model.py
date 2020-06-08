from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
from .train import get_model, get_optimizer, get_loss_function, get_compression
import torch.utils.data as data
from . import dataset

class SpeechClassifierModule(LightningModule):

    def __init__(self, config):
        super().__init__()

        # TODO lit logger to saves hparams (also outdated to use) which causes error TypeError: can't pickle int objects
        self.hparams = config

        # trainset needed to set values in hparams
        self.train_set, self.dev_set, self.test_set = self.hparams["dataset_cls"].splits(self.hparams) 
        self.hparams["width"] = self.train_set.width
        self.hparams["height"] = self.train_set.height
        
        self.model = get_model(self.hparams)
        self.criterion = get_loss_function(self.model, self.hparams)
        self.optimizer = get_optimizer(self.hparams, self.model)
        self.compression_scheduler = get_compression(config,self.model,self.optimizer)
        self.train_set, self.dev_set, self.test_set = config["dataset_cls"].splits(config)


    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        if self.compression_scheduler is not None:
                self.compression_scheduler.on_minibatch_begin(self.current_epoch)

        if self.compression_scheduler is not None:
                self.compression_scheduler.on_minibatch_end(self.current_epoch)
        
    def configure_optimizers(self):
        return self.optimizer
        
    def train_dataloader(self):
        
        train_batch_size = self.hparams["batch_size"]
        collate_fn = dataset.ctc_collate_fn #if train_set.loss_function == "ctc" else None
        return data.DataLoader(self.train_set,
                                batch_size=train_batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=self.hparams["num_workers"], 
                                collate_fn=collate_fn)



    # Callbacks
    def on_epoch_start(self):
        if self.compression_scheduler is not None:
                self.compression_scheduler.on_epoch_begin(self.current_epoch)


    def on_batch_end(self):
        if self.compression_scheduler is not None:
                self.compression_scheduler.on_minibatch_end(self.current_epoch)

    def on_epoch_end(self):
        if self.compression_scheduler is not None:
                self.compression_scheduler.on_epoch_end(self.current_epoch)