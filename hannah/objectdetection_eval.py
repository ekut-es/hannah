import logging


from pathlib import Path

import hydra
import torch

from hannah.datasets.Kitti import Kitti, object_collate_fn

from hydra.utils import to_absolute_path, instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import reset_seed, seed_everything


def eval_train(module):
    trainer = Trainer(gpus=1, deterministic=True)
    trainer.validate(model=module, ckpt_path=None)
    reset_seed()
    trainer.test(model=module, ckpt_path=None)


def eval_steps(config, module, hparams, checkpoint):
    methods = config["methods"]

    if "original" in methods:
        module.augmentation.setEvalAttribs(val_pct=0)
        eval_train(module)

    if "full_augmented" in methods:
        module.augmentation.setEvalAttribs(val_pct=100, wait=True, out=True)
        eval_train(module)

    if "real_rain" in methods:
        folder = hparams["dataset"]["kitti_folder"]
        hparams["dataset"]["kitti_folder"] = folder[: folder.rfind("/")] + "/real_rain"
        hparams["dataset"]["test_pct"] = 50
        hparams["dataset"]["dev_pct"] = 50
        real_module = instantiate(hparams)
        real_module.setup("test")
        real_module.load_state_dict(checkpoint["state_dict"])
        real_module.augmentation.setEvalAttribs(val_pct=0)
        eval_train(real_module)


def eval_checkpoint(config: DictConfig, checkpoint):
    seed_everything(1234, workers=True)
    checkpoint_path = to_absolute_path(checkpoint)

    # FIXME: remove when snapshots use new modules
    import sys
    import hannah

    sys.modules["speech_recognition"] = hannah
    ##

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    hparams = checkpoint["hyper_parameters"]

    if "_target_" not in hparams:
        target = "hannah.modules.object_detection.ObjectDetectionModule"
        logging.warning("Target class not given in checkpoint assuming: %s", target)
        hparams["_target_"] = target

    hparams["num_workers"] = 0
    hparams["augmentation"] = config["augmentation"]
    module = instantiate(hparams)
    module.setup("test")
    module.load_state_dict(checkpoint["state_dict"])

    eval_steps(config, module, hparams, checkpoint)


def eval(config: DictConfig):
    checkpoints = config.checkpoints
    if isinstance(checkpoints, str):
        checkpoints = [checkpoints]

    if not checkpoints:
        logging.error(
            "Please give at least one path for model checkpoints checkpoints=[<checkpoint path>]"
        )
        return False

    for checkpoint in checkpoints:
        eval_checkpoint(config, checkpoint)


@hydra.main(config_name="objectdetection_eval", config_path="conf")
def main(config: DictConfig):
    return eval(config)


if __name__ == "__main__":
    main()
