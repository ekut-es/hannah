import logging
from pathlib import Path
from typing import Any, Optional, Type

import tabulate
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import reset_seed, seed_everything

import hannah.modules.classifier
import hydra


def eval_checkpoint(config: DictConfig, checkpoint) -> None:
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

    trainer = Trainer(gpus=1, deterministic=True)
    # reset_seed()
    # trainer.validate(model=module, ckpt_path=None)
    # reset_seed()
    # trainer.test(model=module, ckpt_path=None)

    snr_values = []
    for test_snr in [-5.0, 0, 5, 10, 15, 20, 25, 30, 35, 40]:
        hparams["dataset"]["test_snr"] = test_snr
        hparams["num_workers"] = 8
        module = instantiate(hparams, _recursive_=False)
        module.setup("test")
        module.load_state_dict(checkpoint["state_dict"])

        reset_seed()
        trainer.test(model=module, ckpt_path=None)
        metric = module.test_metrics["test_accuracy"].compute()
        snr_values.append((test_snr, metric))

    print(tabulate.tabulate(snr_values, headers=["SNR", "Accuracy"]))


def eval(config: DictConfig) -> Optional[bool]:
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
    return eval(config)
