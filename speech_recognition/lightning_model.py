from pytorch_lightning.core.lightning import LightningModule

# from pytorch_lightning.metrics.functional import (
#     accuracy,
#     f1_score,
#     recall,
# )
from .train import (
    get_loss_function,
    get_optimizer,
    get_model,
    save_model,
    get_lr_scheduler,
)

import torch
import torch.utils.data as data
from .dataset import ctc_collate_fn, SpeechDataset
from .utils import _locate, config_pylogger
from pytorch_lightning.metrics import Accuracy, Recall
from pytorch_lightning.metrics.functional import f1_score


class SpeechClassifierModule(LightningModule):
    def __init__(self, config, log_dir, msglogger):
        super().__init__()
        # torch.autograd.set_detect_anomaly(True)
        # TODO lit logger to saves hparams (also outdated to use)
        # which causes error TypeError: can't pickle int objects
        self.hparams = config

        # model
        self.train_set, self.dev_set, self.test_set = _locate(
            config["dataset_cls"]
        ).splits(config)
        self.hparams["width"] = self.train_set.width
        self.hparams["height"] = self.train_set.height
        self.model = get_model(self.hparams)

        # loss function
        self.criterion = get_loss_function(self.model, self.hparams)

        # logging
        self.log_dir = log_dir
        self.msglogger = msglogger
        self.msglogger.info("speech classifier initialized")

        # summarize model architecture
        dummy_width, dummy_height = self.train_set.width, self.train_set.height
        dummy_input = torch.zeros(1, dummy_height, dummy_width)
        self.example_input_array = dummy_input
        self.bn_frozen = False

        # metrics
        self.prepare_metrics()

    # PREPARATION
    def configure_optimizers(self):
        optimizer = get_optimizer(self.hparams, self)
        scheduler = get_lr_scheduler(self.hparams, optimizer)

        return [optimizer], [scheduler]

    def prepare_metrics(self):

        # in case of branchy nets declare multiple objects per metric
        if hasattr(self.model, "n_pieces"):
            for idx in range(self.model.n_pieces):

                # accuracy
                acc_str = f"self.accuracy_branch_{idx}"
                code_str = f"{acc_str} = {Accuracy()}"
                exec(code_str)

                # recall
                recall_str = f"self.recall_branch_{idx}"
                code_str = f"{recall_str} = {Recall()}"
                exec(code_str)
        else:
            self.accuracy = Accuracy()
            self.recall = Recall()

    def get_batch_metrics(self, output, y, loss, prefix):

        # in case of multiple outputs
        if isinstance(output, list):
            # log for each output
            for idx, out in enumerate(output):
                # accuracy
                log_acc_str = f"self.log('{prefix}_branch_{idx}_acc_step', self.accuracy_branch_{idx}(out,y))"
                exec(log_acc_str)

                # recall
                log_recall_str = f"self.log('{prefix}_branch_{idx}_recall_step', self.recall_branch_{idx}(out,y))"
                exec(log_recall_str)

                # TODO f1_score too?

        else:
            self.log("train_acc_step", self.accuracy(output, y))
            self.log("train_recall_step", self.recall(output, y))
            self.log(
                "train_f1",
                f1_score(output, y),
                on_step=True,
                on_epoch=True,
                logger=True,
            )

        # only one loss allowed
        # also in case of branched networks
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)

    def get_epoch_metrics(self, prefix):
        # log epoch metric
        if hasattr(self.model, "n_pieces"):
            for idx in range(self.model.n_pieces):
                # accuracy
                log_acc_str = f"self.log('{prefix}_branch_{idx}_acc_epoch', self.accuracy_branch_{idx}.compute())"
                exec(log_acc_str)

                # recall
                log_recall_str = f"self.log('{prefix}_branch_{idx}_recall_epoch', self.recall_branch_{idx}.compute())"
                exec(log_recall_str)

        else:
            self.log(f"{prefix}_acc_epoch", self.accuracy.compute())
            self.log(f"{prefix}_recall_epoch", self.recall.compute())

    # TRAINING CODE
    def training_step(self, batch, batch_idx):

        x, x_len, y, y_len = batch
        output = self(x)
        y = y.view(-1)
        loss = self.criterion(output, y)

        # --- after loss
        for callback in self.trainer.callbacks:
            if hasattr(callback, "on_before_backward"):
                callback.on_before_backward(self.trainer, self, loss)
        # --- before backward

        # METRICS
        self.get_batch_metrics(output, y, loss, "train")

        return loss

    def training_epoch_end(self, outs):
        # log epoch metrics
        self.get_epoch_metrics("train")

    def train_dataloader(self):

        train_batch_size = self.hparams["batch_size"]
        train_loader = data.DataLoader(
            self.train_set,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.hparams["num_workers"],
            collate_fn=ctc_collate_fn,
        )

        self.batches_per_epoch = len(train_loader)

        return train_loader

    # VALIDATION CODE

    def validation_step(self, batch, batch_idx):

        # dataloader provides these four entries per batch
        x, x_length, y, y_length = batch

        # INFERENCE
        output = self.model(x)
        y = y.view(-1)
        loss = self.criterion(output, y)

        # METRICS
        # batch_acc, batch_f1, batch_recall = self.get_batch_metrics(output, y)
        # result = EvalResult(loss)
        # log_vals = {
        #     "val_loss": loss,
        #     "val_acc": batch_acc,
        #     "val_f1": batch_f1,
        #     "val_recall": batch_recall,
        # }
        # # TODO sync across devices in case of multi gpu via kwarg sync_dist=True
        # result.log_dict(log_vals)

        return loss

    def val_dataloader(self):

        dev_loader = data.DataLoader(
            self.dev_set,
            batch_size=min(len(self.dev_set), 16),
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            collate_fn=ctc_collate_fn,
        )

        return dev_loader

    # TEST CODE
    def test_step(self, batch, batch_idx):

        # dataloader provides these four entries per batch
        x, x_length, y, y_length = batch

        output = self.model(x)
        y = y.view(-1)
        loss = self.criterion(output, y)

        # METRICS
        # batch_acc, batch_f1, batch_recall = self.get_batch_metrics(output, y)
        # result = EvalResult(loss)
        # log_vals = {
        #     "test_loss": loss,
        #     "test_acc": batch_acc,
        #     "test_f1": batch_f1,
        #     "test_recall": batch_recall,
        # }
        # # TODO sync across devices in case of multi gpu via kwarg sync_dist=True
        # result.log_dict(log_vals)

        return loss

    def test_dataloader(self):

        test_loader = data.DataLoader(
            self.test_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            collate_fn=ctc_collate_fn,
        )

        return test_loader

    # FORWARD (overwrite to train instance of this class directly)
    def forward(self, x):
        return self.model(x)

    # CALLBACKS
    def on_train_end(self):
        # TODO currently custom save, in future proper configure lighting for saving ckpt
        save_model(
            self.log_dir,
            self.model,
            self.test_set,
            config=self.hparams,
            msglogger=self.msglogger,
        )
