import logging


from pathlib import Path

import hydra
import torch


from hydra.utils import to_absolute_path, instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import reset_seed, seed_everything

import speech_recognition.modules.classifier


def eval_checkpoint(config: DictConfig, checkpoint):
    seed_everything(1234, workers=True)
    checkpoint_path = to_absolute_path(checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    hparams = checkpoint["hyper_parameters"]

    if "_target_" not in hparams:
        target = "speech_recognition.modules.classifier.StreamClassifierModule"
        logging.warning("Target class not given in checkpoint assuming: %s", target)
        hparams["_target_"] = target

    hparams["num_workers"] = 8
    module = instantiate(hparams)
    module.setup("test")
    module.load_state_dict(checkpoint["state_dict"])

    trainer = Trainer(gpus=1, deterministic=True)
    trainer.validate(model=module, ckpt_path=None)
    reset_seed()
    trainer.test(model=module, ckpt_path=None)


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


@hydra.main(config_name="eval", config_path="conf")
def main(config: DictConfig):
    return eval(config)
