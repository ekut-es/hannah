import logging

from .classifier import ClassifierModule
from .config_utils import get_loss_function, get_model

try:
    from pycocotools.cocoeval import COCOeval
except ModuleNotFoundError:
    COCOeval = None


import torch
import torch.utils.data as data
from hydra.utils import get_class, instantiate

from hannah.datasets.Kitti import object_collate_fn
from hannah.modules.augmentation.augmentation import Augmentation
from hannah.modules.augmentation.bordersearch import (
    Bordersearch,
    Opts,
    dut_fun,
    random_sample,
)


class ObjectDetectionModule(ClassifierModule):
    def __init__(self, augmentation: list(), *args, **kwargs):
        self.augmentation = Augmentation(augmentation)
        self.borderparams = self.augmentation.fillParams()
        self.first_step = True
        super().__init__(*args, **kwargs)

        if COCOeval is None:
            self.msglogger.error("Could not find cocotools")
            self.msglogger.error(
                "please install with poetry install -E object-detection"
            )

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

            self.num_classes = len(self.train_set.class_names) - 1

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
                _recursive_=False,
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

    def bordersearch(self):
        print("################# Bordersearch starts #################")
        brs = Bordersearch()
        opts = Opts(
            parameters=self.borderparams,
            runs=1,
            augmentation_conf=self.augmentation.conf,
            best_path=self.trainer.checkpoint_callback.best_model_path,
        )
        opts.dut_fun = dut_fun
        opts.sample_fun = random_sample
        conf = brs.find_waterlevel(opts, self.augmentation.waterlevel)
        self.augmentation.changeParams(self.borderparams, conf)
        print("################# Bordersearch ends #################")

    def train_dataloader(self):

        if (
            self.trainer.current_epoch != 0
            and self.augmentation.pct != 0
            and (self.trainer.current_epoch % self.augmentation.bordersearch_epochs)
            == 0
        ):
            self.bordersearch()

        train_batch_size = self.hparams["batch_size"]
        dataset_conf = self.hparams.dataset
        sampler = None
        sampler_type = dataset_conf.get("sampler", "random")
        if sampler_type == "weighted":
            sampler = self.get_balancing_sampler(self.train_set)
        else:
            sampler = data.RandomSampler(self.train_set)

        self.augmentation.augment(self.train_set)
        train_loader = data.DataLoader(
            self.train_set,
            batch_size=min(len(self.train_set), train_batch_size),
            drop_last=True,
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
        if self.augmentation.pct != 0 and self.augmentation.val_pct != 0:
            if self.first_step:
                self.augmentation.setEvalAttribs(val_pct=100, wait=True)
                self.augmentation.augment(self.dev_set)
                self.first_step = False
            self.augmentation.setEvalAttribs(reaugment=False)

        self.augmentation.augment(self.dev_set)
        dev_loader = data.DataLoader(
            self.dev_set,
            batch_size=min(len(self.dev_set), 9),
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            collate_fn=object_collate_fn,
            multiprocessing_context="fork" if self.hparams["num_workers"] > 0 else None,
        )

        if self.augmentation.pct != 0 and self.augmentation.val_pct != 0:
            self.augmentation.setEvalAttribs()

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

        return test_loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        cocoGt = self.dev_set.getCocoGt()
        cocoGt.createIndex()

        output = self(x)
        cocoDt = self.model.transformOutput(cocoGt, output, x, y)
        cocoGt.saveImg(cocoDt, y)
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
        cocoGt.clearBatch()

    # TRAINING CODE
    def training_step(self, batch, batch_idx):
        x, y = batch

        output = self.model(x, y)
        loss = sum(output.values())

        metric = dict()
        metric["training_loss"] = loss

        self.log(
            name="training_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, batch, batch_idx):

        # dataloader provides these four entries per batch
        x, y = batch
        cocoGt = self.test_set.getCocoGt()
        cocoGt.createIndex()

        output = self(x)
        cocoDt = self.model.transformOutput(cocoGt, output, x, y)
        cocoGt.saveImg(cocoDt, y)
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
        cocoGt.clearBatch()

    def save(self):
        logging.warning("Onnx export currently not supported for detection modules")
