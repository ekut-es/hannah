import logging
import os
import json
import copy
import platform

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, LoggerCollection
from pytorch_lightning.metrics.metric import MetricCollection
from torch._C import Value
from .config_utils import get_loss_function, get_model
from typing import Optional

from .classifier import ClassifierModule

from pycocotools.cocoeval import COCOeval

import torchvision

from speech_recognition.datasets.Kitti import object_collate_fn
from speech_recognition.datasets.Kitti import KittiCOCO

import torch
import torch.utils.data as data
from hydra.utils import instantiate, get_class


class ObjectDetectionModule(ClassifierModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_data(self):
        pass

    def setup(self, stage):
        # TODO stage variable is not used!
        self.msglogger.info("Setting up model")
        if self.logger:
            self.logger.log_hyperparams(self.hparams)

        if self.initialized:
            return

        self.initialized = True

        if self.hparams.dataset is not None:

            # trainset needed to set values in hparams
            self.train_set, self.dev_set, self.test_set = get_class(
                self.hparams.dataset.cls
            ).splits(self.hparams.dataset)

            self.num_classes = len(self.train_set.class_names)

        # Create example input
        self.example_input_array = torch.zeros(
            1, 3, self.train_set.img_size[0], self.train_set.img_size[1]
        )

        self.example_feature_array = self.example_input_array

        if hasattr(self.hparams.model, "_target_") and self.hparams.model._target_:
            print(self.hparams.model._target_)
            self.model = instantiate(
                self.hparams.model,
                input_shape=self.example_feature_array.shape,
                labels=self.num_classes,
            )
        else:
            self.hparams.model.width = self.example_feature_array.size(2)
            self.hparams.model.height = self.example_feature_array.size(1)
            self.hparams.model.n_labels = self.num_classes
            self.model = get_model(self.hparams.model)

        # loss function
        self.criterion = get_loss_function(self.model, self.hparams)

    def forward(self, x):
        x = self.model(x)
        return x

    def train_dataloader(self):
        train_batch_size = self.hparams["batch_size"]
        dataset_conf = self.hparams.dataset
        sampler = None
        sampler_type = dataset_conf.get("sampler", "random")
        if sampler_type == "weighted":
            sampler = self.get_balancing_sampler(self.train_set)
        else:
            sampler = data.RandomSampler(self.train_set)

        train_loader = data.DataLoader(
            self.train_set,
            batch_size=min(len(self.train_set), train_batch_size),
            drop_last=True,
            pin_memory=True,
            num_workers=self.hparams["num_workers"],
            collate_fn=object_collate_fn,
            sampler=sampler,
            multiprocessing_context="fork" if self.hparams["num_workers"] > 0 else None,
        )

        # if self.device.type == "cuda":
        #    train_loader = AsynchronousLoader(train_loader, device=self.device)

        self.batches_per_epoch = len(train_loader)

        return train_loader

    def val_dataloader(self):

        dev_loader = data.DataLoader(
            self.dev_set,
            batch_size=min(len(self.dev_set), 9),
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            collate_fn=object_collate_fn,
            multiprocessing_context="fork" if self.hparams["num_workers"] > 0 else None,
        )

        # if self.device.type == "cuda":
        #    dev_loader = AsynchronousLoader(dev_loader, device=self.device)

        return dev_loader

    def test_dataloader(self):
        test_loader = data.DataLoader(
            self.test_set,
            batch_size=min(len(self.test_set), 9),
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            collate_fn=object_collate_fn,
            multiprocessing_context="fork" if self.hparams["num_workers"] > 0 else None,
        )

        # if self.device.type == "cuda":
        #    test_loader = AsynchronousLoader(test_loader, device=self.device)

        return test_loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        cocoGt = self.test_set.getCocoGt()
        cocoGt.createIndex()

        output = self(x)
        cocoDt = cocoGt.transformOutput(output)
        cocoDt = cocoGt.loadRes(cocoDt)
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        metric = dict()
        metric["val_ap"] = cocoEval.stats[0].item()
        metric["val_ap_75"] = cocoEval.stats[2].item()
        metric["val_ar"] = cocoEval.stats[6].item()
        metric["val_ar_100dets"] = cocoEval.stats[8].item()

        self.log_dict(metric, on_step=False, on_epoch=True, prog_bar=True)

    # TRAINING CODE
    def training_step(self, batch, batch_idx):
        x, y = batch

        output = self.model(x, y)
        loss = sum(output.values())

        return loss

    def test_step(self, batch, batch_idx):

        # dataloader provides these four entries per batch
        x, y = batch
        cocoGt = self.test_set.getCocoGt()
        cocoGt.createIndex()

        output = self(x)
        cocoDt = cocoGt.transformOutput(output)
        cocoDt = cocoGt.loadRes(cocoDt)
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        metric = dict()
        metric["test_ap"] = cocoEval.stats[0].item()
        metric["test_ap_75"] = cocoEval.stats[2].item()
        metric["test_ar"] = cocoEval.stats[6].item()
        metric["test_ar_100dets"] = cocoEval.stats[8].item()

        self.log_dict(metric, on_step=False, on_epoch=True, prog_bar=True)
