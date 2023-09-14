#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import reset_seed, seed_everything

from hannah.datasets.Kitti import Kitti


def eval_train(config, module, test=True):
    """

    Args:
      config: param module:
      test: Default value = True)
      module:

    Returns:

    """
    gpus = config["gpus"] if len(config["gpus"]) > 0 else 0
    trainer = Trainer(gpus=gpus, deterministic=True, logger=False)
    val = trainer.validate(model=module, ckpt_path=None, verbose=test)

    return val


def eval_steps(config, module, hparams, checkpoint):
    """

    Args:
      config: param module:
      hparams: param checkpoint:
      module:
      checkpoint:

    Returns:

    """
    methods = config["methods"]
    folder = hparams["dataset"]["kitti_folder"]

    retval = dict()
    if "original" in methods:
        module.augmentation.setEvalAttribs(val_pct=0)
        hparams["dataset"]["dev_pct"] = 100
        retval["original"] = eval_train(config, module)

    if "full_augmented" in methods:
        module.augmentation.setEvalAttribs(val_pct=100, wait=True, out=True)
        hparams["dataset"]["dev_pct"] = 100
        retval["full_augmented"] = eval_train(config, module)

    if "real_rain" in methods:
        hparams["dataset"]["kitti_folder"] = folder[: folder.rfind("/")] + "/real_rain"
        hparams["dataset"]["dev_pct"] = 100
        real_module = instantiate(hparams, _recursive_=False)
        real_module.setup("test")
        real_module.load_state_dict(checkpoint["state_dict"])
        real_module.augmentation.setEvalAttribs(val_pct=0)
        retval["real_rain"] = eval_train(config, real_module)

    if "dawn_rain" in methods:
        hparams["dataset"]["kitti_folder"] = folder[: folder.rfind("/")] + "/DAWN/Rain"
        hparams["dataset"]["dev_pct"] = 100
        real_module = instantiate(hparams)
        real_module.setup("test")
        real_module.load_state_dict(checkpoint["state_dict"])
        real_module.augmentation.setEvalAttribs(val_pct=0)
        retval["dawn_rain"] = eval_train(config, real_module)

    if "dawn_snow" in methods:
        hparams["dataset"]["kitti_folder"] = folder[: folder.rfind("/")] + "/DAWN/Snow"
        hparams["dataset"]["dev_pct"] = 100
        real_module = instantiate(hparams)
        real_module.setup("test")
        real_module.load_state_dict(checkpoint["state_dict"])
        real_module.augmentation.setEvalAttribs(val_pct=0)
        retval["dawn_snow"] = eval_train(config, real_module)

    if "dawn_fog" in methods:
        hparams["dataset"]["kitti_folder"] = folder[: folder.rfind("/")] + "/DAWN/Fog"
        real_module = instantiate(hparams)
        real_module.setup("test")
        real_module.load_state_dict(checkpoint["state_dict"])
        real_module.augmentation.setEvalAttribs(val_pct=0)
        retval["dawn_fog"] = eval_train(config, real_module)

    if "bordersearch" in methods:
        module.augmentation.setEvalAttribs(val_pct=100, reaugment=False)
        retval["bordersearch"] = eval_train(config, module, False)

    return retval


def eval_checkpoint(config: DictConfig, checkpoint):
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
    module.first_step = False

    return eval_steps(config, module, hparams, checkpoint)


def eval(config: DictConfig):
    """

    Args:
      config: DictConfig:
      config: DictConfig:
      config: DictConfig:

    Returns:

    """
    retval = list()
    checkpoints = (
        config.checkpoints if hasattr(config, "checkpoints") else config["checkpoints"]
    )
    if isinstance(checkpoints, str):
        checkpoints = [checkpoints]

    if not checkpoints:
        logging.error(
            "Please give at least one path for model checkpoints checkpoints=[<checkpoint path>]"
        )
        return False

    for checkpoint in checkpoints:
        retval.append(eval_checkpoint(config, checkpoint))
    return retval


@hydra.main(
    config_name="objectdetection_eval", config_path="../conf", version_base="1.2"
)
def main(config: DictConfig):
    """

    Args:
      config: DictConfig:
      config: DictConfig:
      config: DictConfig:

    Returns:

    """
    return eval(config)


if __name__ == "__main__":
    main()
