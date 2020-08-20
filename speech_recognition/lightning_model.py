from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy, confusion_matrix, f1_score, recall
from .train import get_loss_function, get_optimizer, get_model, save_model
import torch.utils.data as data
import torch
from . import dataset
import numpy as np
from .utils import _locate, config_pylogger
import distiller
import torchnet.meter as tnt


class SpeechClassifierModule(LightningModule):
    def __init__(self, model_name, config, log_dir):
        super().__init__()

        # TODO lit logger to saves hparams (also outdated to use)
        # which causes error TypeError: can't pickle int objects
        self.hparams = config

        # trainset needed to set values in hparams
        self.train_set, self.dev_set, self.test_set = _locate(config["dataset_cls"]).splits(config)
        self.hparams["width"] = self.train_set.width
        self.hparams["height"] = self.train_set.height
        self.compression_scheduler = None  # initialize on train start
        self.model = get_model(self.hparams)
        self.criterion = get_loss_function(self.model, self.hparams)
        self.optimizer = get_optimizer(self.hparams, self.model)
        self.log_dir = log_dir
        self.collate_fn = dataset.ctc_collate_fn  # if train_set.loss_function == "ctc" else None
        self.msglogger = config_pylogger('logging.conf', "lightning-logger", self.log_dir)
        self.msglogger.info("speech classifier initialized")

    # PREPARATION
    def configure_optimizers(self):
        return self.optimizer

    def get_batch_metrics(self, output, y):

        y = y.view(-1) 
        self.loss = self.criterion(output, y)

        output_max =  output.argmax(dim=1)
        batch_acc = accuracy(output_max, y, self.hparams['n_labels'])
        batch_f1 = f1_score(output_max, y)
        batch_recall = recall(output_max, y)

        return batch_acc, batch_f1, batch_recall

    def get_epoch_metrics(self, outputs, phase):

        # respecting the naming specification of lightning
        if phase == "train":
            loss_key = "loss"
        else:
            loss_key = f'{phase}_loss'

        avg_loss = torch.stack([x[loss_key] for x in outputs]).mean()

        n_outputs = len(outputs)
        acc_mean = 0
        f1_mean = 0
        recall_mean = 0

        confusion = tnt.ConfusionMeter(self.hparams["n_labels"])

        for output in outputs:
            acc_mean += output[f'{phase}_batch_acc']
            f1_mean += output[f'{phase}_batch_f1']
            recall_mean += output[f'{phase}_batch_recall']
            curr_output = output[f'{phase}_output'].data
            curr_labels = output[f'{phase}_y'].view(-1)
            confusion.add(curr_output, curr_labels)

        acc_mean /= n_outputs
        f1_mean /= n_outputs
        recall_mean /= n_outputs

        # 'prettier' printing
        np.set_printoptions(suppress=True)
        print("-- loss: %0.3f" % (avg_loss))
        print("-- accuracy: %0.3f" % (acc_mean))
        print("-- f1: %0.3f" % (f1_mean))
        print("-- recall: %0.3f" % (recall_mean))
        print(f"-- confusion:\n{confusion.value()}")

        return avg_loss, acc_mean, f1_mean, recall_mean

    ### TRAINING CODE ###

    def training_step(self, batch, batch_idx):

        self.batch_idx = batch_idx

        if self.compression_scheduler is not None:
            self.compression_scheduler.on_minibatch_begin(self.current_epoch, batch_idx, self.batches_per_epoch)

        x, x_len, y, y_len = batch
        output = self(x)
        y = y.view(-1) 

        if self.compression_scheduler is not None:
            self.compression_scheduler.on_minibatch_end(self.current_epoch, batch_idx, self.batches_per_epoch)

        # METRICS
        batch_acc, batch_f1, batch_recall = self.get_batch_metrics(output, y)

        results = {
            # output directory
            'loss': self.loss,  # naming specification of lightning
            'train_output': output,
            'train_y': y,
            'train_batch_acc': batch_acc,
            'train_batch_f1': batch_f1,
            'train_batch_recall': batch_recall,
            # log directory
            'log': {
                'train_loss': self.loss,
                'train_batch_acc': batch_acc,
                'train_batch_f1': batch_f1,
                'train_batch_recall': batch_recall
                },
            # progress bar
            'progress_bar': {'train_batch_acc': batch_acc}
        }

        return results

    def training_epoch_end(self, outputs):
        print("\n")
        print("Training:")
        avg_loss, acc_mean, f1_mean, recall_mean = self.get_epoch_metrics(outputs, "train")

        # logs
        results = {
            'log': {
                'train_loss_mean': avg_loss.item(),
                'train_acc_mean': acc_mean.item(),
                'train_f1_mean': f1_mean.item(),
                'train_recall_mean': recall_mean.item()
            },
        }

        return results

    def train_dataloader(self):

        train_batch_size = self.hparams["batch_size"]
        train_loader = data.DataLoader(
                                self.train_set,
                                batch_size=train_batch_size,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                num_workers=self.hparams["num_workers"],
                                collate_fn=self.collate_fn)

        self.batches_per_epoch = len(train_loader)

        return train_loader

    ### VALIDATION CODE ###

    def validation_step(self, batch, batch_idx):

        # dataloader provides these four entries per batch
        x, x_length, y, y_length = batch

        # INFERENCE
        output = self.model(x)

        # METRICS
        batch_acc, batch_f1, batch_recall= self.get_batch_metrics(output, y)

        # RESULT DICT
        results = {
            # output directory
            'val_loss': self.loss,  # mandatory
            'val_output': output,
            'val_y': y,
            'val_batch_acc': batch_acc,
            'val_batch_f1': batch_f1,
            'val_batch_recall': batch_recall,
            # log directory
            'log': {
                'val_loss': self.loss,
                'val_batch_acc': batch_acc,
                'val_batch_f1': batch_f1,
                'val_batch_recall': batch_recall
            }
        }
        return results

    def validation_epoch_end(self, outputs):
        print("\n")
        print("Validation:")
        avg_loss, acc_mean, f1_mean, recall_mean = self.get_epoch_metrics(outputs, "val")

        results = {
            'log': {
                'val_loss_mean': avg_loss.item(),
                'val_acc_mean': acc_mean,
                'val_f1_mean': f1_mean,
                'val_recall_mean': recall_mean
            }
        }
        return results

    def val_dataloader(self):

        dev_loader = data.DataLoader(
                                self.dev_set,
                                batch_size=min(len(self.dev_set), 16),
                                shuffle=False,
                                num_workers=self.hparams["num_workers"],
                                collate_fn=self.collate_fn)

        return dev_loader

    ### TEST CODE ###

    def test_step(self, batch, batch_idx):

        # dataloader provides these four entries per batch
        x, x_length, y, y_length = batch

        output = self.model(x)

        # METRICS
        batch_acc, batch_f1, batch_recall = self.get_batch_metrics(output, y)

        # RESULT DICT
        results = {
            # output directory
            'test_loss': self.loss,  # mandatory
            'test_output': output,
            'test_y': y,
            'test_batch_acc': batch_acc,
            'test_batch_f1': batch_f1,
            'test_batch_recall': batch_recall,
            # log directory
            'log': {
                'test_loss': self.loss,
                'test_batch_acc': batch_acc,
                'test_batch_f1': batch_f1,
                'test_batch_recall': batch_recall
            }
        }
        return results

    def test_epoch_end(self, outputs):

        print("\n")
        print("Test:")
        avg_loss, acc_mean, f1_mean, recall_mean = self.get_epoch_metrics(outputs, "test")

        results = {
            'log': {
                'test_loss_mean': avg_loss.item(),
                'test_acc_mean': acc_mean,
                'test_f1_mean': f1_mean,
                'test_recall_mean': recall_mean
            }
        }
        return results

    def test_dataloader(self):

        test_loader = data.DataLoader(
                                    self.test_set,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=self.hparams["num_workers"],
                                    collate_fn=self.collate_fn)

        return test_loader

    # FORWARD (overwrite to train instance of this class directly)
    def forward(self, x):
        return self.model(x)

    # CALLBACKS
    def on_train_start(self):
        if self.hparams["compress"]:
            self.model.to(self.device)
            # msglogger.info("Activating compression scheduler")
            self.compression_scheduler = distiller.file_config(
                                                        self.model,
                                                        self.optimizer,
                                                        self.hparams["compress"])

    def on_epoch_start(self):
        if self.compression_scheduler is not None:
            self.compression_scheduler.on_epoch_begin(self.current_epoch)

    def on_batch_end(self):
        if self.compression_scheduler is not None:
            self.compression_scheduler.before_backward_pass(
                                                    self.current_epoch,
                                                    self.batch_idx,
                                                    self.batches_per_epoch,
                                                    self.loss)

    def on_epoch_end(self):
        if self.compression_scheduler is not None:
            self.compression_scheduler.on_epoch_end(self.current_epoch)

    def on_train_end(self):
        # TODO currently custom save, in future proper configure lighting for saving ckpt
        save_model(self.log_dir, self.model, self.test_set, config=self.hparams, msglogger=self.msglogger)
