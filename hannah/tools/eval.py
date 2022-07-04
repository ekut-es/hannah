import logging
from pathlib import Path
from typing import Any, Optional, Type

import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import reset_seed, seed_everything

import hannah.modules.classifier
import hydra


def eval_checkpoint(config: DictConfig, checkpoint) -> None:
    """

    Args:
      config: DictConfig:
      checkpoint:
      config: DictConfig:
      config: DictConfig:

    Returns:

    """
    seed_everything(1234, workers=True)
    checkpoint_path = to_absolute_path(checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    hparams = checkpoint["hyper_parameters"]
    if "_target_" not in hparams:
        target = config.default_target
        logging.warning("Target class not given in checkpoint assuming: %s", target)
        hparams["_target_"] = target

    hparams["num_workers"] = 8
    module = instantiate(hparams, _recursive_=False)
    module.setup("test")
    module.load_state_dict(checkpoint["state_dict"])

    trainer = Trainer(gpus=0, deterministic=True)
    trainer.validate(model=module, ckpt_path=None)
    reset_seed()
    trainer.test(model=module, ckpt_path=None)


def eval(config: DictConfig) -> Optional[bool]:
    """

    Args:
      config: DictConfig:
      config: DictConfig:
      config: DictConfig:

    Returns:

    """
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


@hydra.main(config_name="eval", config_path="conf", version_base="1.2")
def main(config: DictConfig):
    """

    Args:
      config: DictConfig:
      config: DictConfig:
      config: DictConfig:

    Returns:

    """
    return eval(config)