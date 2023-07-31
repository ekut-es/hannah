import logging
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
import pickle

try:
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils
    from pcdet.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result
    from pcdet.datasets.kitti.kitti_object_eval_python.kitti_common import get_label_anno

except ImportError:
    print("Module pcdet not found")

from hydra.utils import instantiate
from hannah.datasets.dense_lidar import DenseLidarDataset
from hannah.datasets.kitti_lidar import KittiLidarDataset
from hannah.datasets.scope_lidar import ScopeLidarDataset
from omegaconf import OmegaConf
from easydict import EasyDict

from pointcloud_viewer import PCViewer

msglogger = logging.getLogger(__name__)

Dataset = {
    'KittiDataset': KittiLidarDataset,
    'DenseDataset': DenseLidarDataset,
    'ScopeDataset': ScopeLidarDataset
}


class LidarDetectionModule(LightningModule):
    def __init__(self, *args, **kwargs):
        self.dist_train = False
        super().__init__()
        self.save_hyperparameters()

        self.augmentor = instantiate(self.hparams.augmentor)

        self.train_set = Dataset[self.hparams.dataset.DATASET](dataset_cfg=self.hparams.dataset, class_names=self.hparams.dataset.CLASS_NAMES, training=True, root_path=Path(self.hparams.dataset.DATA_PATH), augmentor=self.augmentor)
        self.val_set = Dataset[self.hparams.dataset.DATASET](dataset_cfg=self.hparams.dataset, class_names=self.hparams.dataset.CLASS_NAMES, training=False, root_path=Path(self.hparams.dataset.DATA_PATH))

        self.model = None

    def setup(self, stage):
        print("Setting up model on device: ", self.device)
        if not self.model:
            model_cfg = EasyDict(OmegaConf.to_object(self.hparams.model.MODEL))
            self.model = build_network(model_cfg=model_cfg, num_class=len(self.hparams.dataset.CLASS_NAMES), dataset=self.train_set)

    def forward(self, x):
        self.load_data_to_gpu(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        ret_dict, tb_dict, disp_dict = self.forward(batch)
        loss = ret_dict["loss"].mean()
        ret_dict["loss"] = loss
        self.log("loss", loss)
        return ret_dict

    def test_step(self, batch, batch_idx):
        pred_dicts, ret_dict = self.forward(batch)
        prediction_dict = self.test_set.generate_prediction_dicts(batch, pred_dicts, self.test_set.class_names)
        return prediction_dict

    def test_epoch_end(self, outputs):
        dt = [det for detections in outputs for det in detections]
        if len(dt) != len(self.test_set) or len(dt) == 0:
            print("size mismatch, skipping eval!")
            return
        ap_str, ap = self.test_set.evaluation(dt, OmegaConf.to_object(self.hparams.dataset.CLASS_NAMES))

        msglogger.info("Test Metrics:\n %s", ap_str)

        self.log("AP", ap)
        self.log("Car_3D_Moderate", ap["Car_3d/moderate_R40"])
        self.log("Car_BEV_Moderate", ap["Car_bev/moderate_R40"])
        self.log("Car_3D_AP", ap["Car_3d/moderate_R40"])

    def validation_step(self, batch, batch_idx):
        if self.trainer.sanity_checking:
            return
        pred_dicts, ret_dict = self.forward(batch)
        prediction_dict = self.val_set.generate_prediction_dicts(batch, pred_dicts, self.val_set.class_names)

        if not self.hparams.dataset.get('SAVE_DETECTIONS', True) or self.hparams.dataset.get('LATE_FUSION', False):
            for idx, prediction in enumerate(prediction_dict):
                prediction['coop_boxes'] = batch['coop_boxes'][idx].cpu().numpy()
                prediction['coop_scores'] = batch['coop_scores'][idx].cpu().numpy()
                prediction['coop_ids'] = batch['coop_ids'][idx].cpu().numpy()
                prediction['coop_vehicle_ids'] = batch['coop_vehicle_ids'][idx].cpu().numpy()

        return prediction_dict

    def validation_epoch_end(self, outputs):
        dt = [det for detections in outputs for det in detections]

        if self.trainer and self.trainer.sanity_checking:
            self.model.train()
            return

        ap_str, ap = self.val_set.evaluation(dt, OmegaConf.to_object(self.hparams.dataset.CLASS_NAMES))
        print(ap_str)

        if self.hparams.dataset.SAVE_DETECTIONS:
            detection_file = self.hparams.dataset.DATA_PATH+self.hparams.dataset.SAVE_DETECTION_VEHICLE+'_DETECTIONS.pkl'
            with open(detection_file, 'wb') as f:
                pickle.dump({'detections': dt, 'AP': ap}, f)

        msglogger.info("Validation Metrics:\n %s", ap_str)

        self.log("AP", ap)
        self.log("Car_3D_AP", ap["Car_3d/moderate_R40"])
        ap_3d_50 = float(ap_str.split(':')[-2].split(',')[0])
        self.log("Car_AP_STR", ap_3d_50)

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        retval = {"optimizer": optimizer}

        if self.hparams.scheduler is not None:
            if self.hparams.scheduler._target_ == "torch.optim.lr_scheduler.OneCycleLR":
                scheduler = instantiate(
                    self.hparams.scheduler,
                    optimizer=optimizer,
                    total_steps=self.total_training_steps(),
                )
                retval["lr_scheduler"] = dict(scheduler=scheduler, interval="step")
            else:
                scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)

                retval["lr_scheduler"] = dict(scheduler=scheduler, interval="epoch")

        return retval

    def total_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps > 0:
            return self.trainer.max_steps

        def ceildiv(a, b):
            return -(a // -b)

        steps_per_epoch = ceildiv(len(self.train_set), self.hparams.batch_size)
        num_devices = max(1, self.trainer.num_devices)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices

        return ceildiv(steps_per_epoch, effective_accum) * self.trainer.max_epochs

    def train_dataloader(self):
        if self.dist_train:
            sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
        else:
            sampler = None

        train_loader = DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
            # shuffle=(sampler is None),
            shuffle=False,
            collate_fn=self.train_set.collate_batch,
            drop_last=False,
            sampler=sampler,
            timeout=0,
            multiprocessing_context="fork" if self.hparams.num_workers > 0 else None
        )

        return train_loader

    def test_dataloader(self):
        if self.dist_train:
            rank, world_size = common_utils.get_dist_info()
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.test_set, world_size, rank, shuffle=False
            )
        else:
            sampler = None

        test_loader = DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=self.test_set.collate_batch,
            drop_last=False,
            sampler=sampler,
            timeout=0,
            multiprocessing_context="fork" if self.hparams.num_workers > 0 else None
        )

        return test_loader

    def val_dataloader(self):
        if self.dist_train:
            rank, world_size = common_utils.get_dist_info()
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_set, world_size, rank, shuffle=False
            )
        else:
            sampler = None
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=self.val_set.collate_batch,
            drop_last=False,
            sampler=sampler,
            timeout=0,
            multiprocessing_context="fork" if self.hparams.num_workers > 0 else None
        )

        return val_loader

    def load_data_to_gpu(self, batch_dict):
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            elif key in ['frame', "frame_id", "metadata", "calib"]:
                continue
            elif key in ["images"]:
                continue
            elif key in ["image_shape"]:
                batch_dict[key] = torch.from_numpy(val).int().to(self.device)
            else:
                batch_dict[key] = torch.from_numpy(val).float().to(self.device)
