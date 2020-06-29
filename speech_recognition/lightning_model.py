from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.metrics.functional import accuracy, confusion_matrix, f1_score, recall
from .train import get_loss_function, get_optimizer, get_model, get_compression, save_model
import torch.utils.data as data
import torch
from . import dataset



from .utils import _locate, config_pylogger


class SpeechClassifierModule(LightningModule):
    def __init__(self, model_name, config, log_dir):
        super().__init__()
        
        # TODO lit logger to saves hparams (also outdated to use) which causes error TypeError: can't pickle int objects
        self.hparams = config

        # trainset needed to set values in hparams
        self.train_set, self.dev_set, self.test_set = _locate(config["dataset_cls"]).splits(config)
        self.hparams["width"] = self.train_set.width
        self.hparams["height"] = self.train_set.height
        
        self.model = get_model(self.hparams)
        self.criterion = get_loss_function(self.model, self.hparams)
        self.optimizer = get_optimizer(self.hparams, self.model)
        self.compression_scheduler = get_compression(config,self.model,self.optimizer)
        self.log_dir = log_dir
        self.collate_fn = dataset.ctc_collate_fn #if train_set.loss_function == "ctc" else None
        self.msglogger = config_pylogger('logging.conf', "lightning-logger", self.log_dir)
        self.msglogger.info("speech classifier initialized")

  
    # PREPARATION
    def configure_optimizers(self):
        return self.optimizer

    ### TRAINING CODE ###
    def train_dataloader(self):
        
        train_batch_size = self.hparams["batch_size"]
        train_loader = data.DataLoader(self.train_set,
                                batch_size=train_batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=self.hparams["num_workers"], 
                                collate_fn=self.collate_fn)
        self.batches_per_epoch = len(train_loader)

        return train_loader

    def training_step(self, batch, batch_idx):
        
        self.batch_idx = batch_idx

        if self.compression_scheduler is not None:
            self.compression_scheduler.on_minibatch_begin(self.current_epoch, batch_idx, self.batches_per_epoch)
    
        x, x_len, y, y_len = batch
        y_hat = self(x)        
        y = y.view(-1)
        
        self.loss = self.criterion(y_hat, y)

        if self.compression_scheduler is not None:
            self.compression_scheduler.on_minibatch_end(self.current_epoch, batch_idx, self.batches_per_epoch)

        # calculates accuracy across all GPUs and all Nodes used in training
        train_batch_acc = accuracy(y_hat, y)
        #train_batch_confusion = confusion_matrix(y_hat, y)
        train_batch_f1 = f1_score(y_hat, y)
        train_batch_recall = recall(y_hat, y)


        results = {
            'loss': self.loss, # mandatory
            'log': {
                'train_loss': self.loss,
                'train_batch_f1' : train_batch_f1,
                'train_batch_recall' : train_batch_recall
                #'train_batch_confusion':train_batch_confusion
                },
            'progress_bar': {'train_batch_acc': train_batch_acc}
        }


        return results
    
    def training_epoch_end(self, outputs):
        n_outputs = len(outputs)
        epoch_acc_mean = 0
        epoch_f1_mean = 0
        epoch_recall_mean = 0

        for output in outputs:
            epoch_acc_mean += output['train_batch_acc']
            #epoch_confusion += output['train_batch_confusion']
            epoch_f1_mean += output['train_batch_f1']
            epoch_recall_mean += output['train_batch_recall']

        epoch_acc_mean /= n_outputs
        epoch_f1_mean /= n_outputs
        epoch_recall_mean /= n_outputs


        print(f"\n epoch {self.current_epoch}:")
        print("accuracy: %0.3f" % (epoch_acc_mean))
        print("f1: %0.3f" % (epoch_f1_mean))
        print("recall: %0.3f" % (epoch_recall_mean))
        #print(f"confusion: {epoch_confusion}")



        # logs
        results = {
            'log': {
                'epochs_acc_mean': epoch_acc_mean.item(),
                'epochs_f1_mean' : epoch_f1_mean.item(),
                'epochs_recall_mean' : epoch_recall_mean.item()
            },
        }

        return results


    ### END TRAINING CODE ###

    ### VALIDATION CODE ###
    def validation_step(self, batch, batch_idx):
        
        # dataloader provides these four entries per batch
        x, x_length, y, y_length = batch

        y_hats = self.model(x)

        if self.hparams["loss"] == "ctc":
                loss = self.criterion(y_hats, y)
        else:    
            y = y.view(-1)
            loss = self.criterion(y_hats, y)

        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def val_dataloader(self):

        dev_loader = data.DataLoader(self.dev_set,
                                 batch_size=min(len(self.dev_set), 16),
                                 shuffle=False,
                                 num_workers=self.hparams["num_workers"],
                                 collate_fn=self.collate_fn)

        return dev_loader

    ### END VALIDATION CODE ###

    ### TEST CODE ###

    def test_step(self, batch, batch_idx):
        
        # dataloader provides these four entries per batch
        x, x_length, y, y_length = batch

        y_hat = self.model(x)

        if self.hparams["loss"] == "ctc":
                loss = self.criterion(y_hat, y)
        else:    
            y = y.view(-1)
            loss = self.criterion(y_hat, y)

        # calculates accuracy across all GPUs and all Nodes used in training
        self.msglogger.info(f"Accuracy: {accuracy(y_hat, y)}")

        return {'test_loss': loss}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def test_dataloader(self):

        test_loader = data.DataLoader(self.test_set,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=self.hparams["num_workers"],
                                 collate_fn=self.collate_fn)

        return test_loader

    ### END TEST CODE ###

    

    # FORWARD (overwrite to train instance of this class directly)
    def forward(self, x):
        return self.model(x)

    # CALLBACKS
    def on_epoch_start(self):
        if self.compression_scheduler is not None:
            self.compression_scheduler.on_epoch_begin(self.current_epoch)

    def on_batch_end(self):
        if self.compression_scheduler is not None:
                self.compression_scheduler.before_backward_pass(self.current_epoch, self.batch_idx,
                                                    self.batches_per_epoch, self.loss)

    def on_epoch_end(self):
        if self.compression_scheduler is not None:
                self.compression_scheduler.on_epoch_end(self.current_epoch)

    def on_train_end(self):
        # TODO currently custom save, in future proper configure lighting for saving ckpt
        save_model(self.log_dir, self.model, self.test_set, config=self.hparams, msglogger=self.msglogger)
        