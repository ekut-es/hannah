#
# Copyright (c) 2024 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
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
import importlib
import logging
import os
import pathlib
import platform
import random
import shutil
import subprocess
import sys
import time
from contextlib import _GeneratorContextManager, contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, List, Type, TypeVar, Union

import hydra
import numpy as np
import nvsmi
import pytorch_lightning
import torch
import torch.nn as nn
from git import InvalidGitRepositoryError, Repo
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    Callback,
    DeviceStatsMonitor,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only


from ..callbacks.clustering import kMeans
from ..callbacks.dump_layers import TestDumperCallback
from ..callbacks.fine_tuning import LinearClassifierTraining
from ..callbacks.optimization import HydraOptCallback
from ..callbacks.pruning import PruningAmountScheduler
from ..callbacks.summaries import FxMACSummaryCallback, MacSummaryCallback
from ..callbacks.svd_compress import SVD


msglogger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def log_execution_env_state() -> None:
    """Log information about the execution environment."""

    logger.info("Environment info:")

    def log_git_state(gitroot):
        """Log the state of the git repository.
        It is useful to know what git tag we're using, and if we have outstanding code.
        """
        try:
            repo = Repo(gitroot)
            assert not repo.bare
        except InvalidGitRepositoryError:
            logger.info(
                "    Cannot find a Git repository.  You probably downloaded an archive"
            )
            return

        if repo.is_dirty():
            logger.info("    Git is dirty")
        try:
            branch_name = repo.active_branch.name
        except TypeError:
            branch_name = "None, Git is in 'detached HEAD' state"
        logger.info("    Active Git branch: %s", branch_name)
        logger.info("    Git commit: %s" % repo.head.commit.hexsha)

    # logger.info("  Number of CPUs: %d", len(os.sched_getaffinity(0)))
    logger.info("  Number of GPUs: %d", torch.cuda.device_count())
    logger.info("  CUDA version: %s", torch.version.cuda)
    logger.info("  CUDNN version: %s", torch.backends.cudnn.version())
    logger.info("  Kernel: %s", platform.release())

    logger.info("  Python: %s", sys.version.replace("\n", "").replace("\r", ""))
    logger.info("  PyTorch: %s", torch.__version__)
    logger.info("  Pytorch Lightning: %s", pytorch_lightning.__version__)
    logger.info("  Numpy: %s", np.__version__)
    logger.info("  Hannah version info:")
    log_git_state(os.path.join(os.path.dirname(__file__), ".."))
    logger.info("  Command line: %s", " ".join(sys.argv))
    logger.info("  ")


def git_version(short=True):
    """Return the current git sha

    Parameters:
        short (bool): If True, return the short (7 character) version of the SHA

    Returns:
        str: The current git SHA
    """

    workdir = os.path.dirname(__file__)

    command = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
    return (
        subprocess.check_output(
            command,
        )
        .decode("utf8")
        .strip()
    )


def common_callbacks(config: DictConfig) -> list:
    callbacks: List[Callback] = []

    lr_monitor = LearningRateMonitor()
    callbacks.append(lr_monitor)

    if config.get("device_stats", None) or config.get("gpu_stats", None):
        if config.get("gpu_stats", None):
            msglogger.warning(
                "config option gpu_stats has been deprecated use device_stats instead"
            )
        device_stats = DeviceStatsMonitor(cpu_stats=config.get("device_stats", False))
        callbacks.append(device_stats)
    use_fx_mac_summary = config.get("fx_mac_summary", False)
    if use_fx_mac_summary:
        mac_summary_callback = FxMACSummaryCallback()
    else:
        mac_summary_callback = MacSummaryCallback()
    callbacks.append(mac_summary_callback)

    if config.get("early_stopping", None):
        stop_callback = hydra.utils.instantiate(config.early_stopping)
        callbacks.append(stop_callback)

    if config.get("dump_test", False):
        callbacks.append(TestDumperCallback())

    if config.get("compression", None):
        config_compression = config.get("compression")
        if config_compression.get("pruning", None):
            pruning_scheduler = PruningAmountScheduler(
                config.compression.pruning.amount, config.trainer.max_epochs
            )
            pruning_config = dict(config.compression.pruning)
            del pruning_config["amount"]
            pruning_callback = hydra.utils.instantiate(
                pruning_config, amount=pruning_scheduler
            )
            callbacks.append(pruning_callback)

        if config_compression.get("decomposition", None):
            compress_after_epoch = config.trainer.max_epochs
            if (
                compress_after_epoch % 2 == 1
            ):  # SVD compression occurs max_epochs/2 epochs. If max_epochs is an odd number, SVD not called
                compress_after_epoch -= 1
            svd = SVD(
                rank_compression=config.compression.decomposition.rank_compression,
                compress_after=compress_after_epoch,
            )
            callbacks.append(svd)

        if config_compression.get("clustering", None):
            kmeans = kMeans(
                cluster=config.compression.clustering.amount,
            )
            callbacks.append(kmeans)

        if config_compression.get("quantization", None):
            quantization_callback = hydra.utils.instantiate(
                config.compression.quantization
            )
            callbacks.append(quantization_callback)

    if config.get("fine_tuning", None):
        fine_tuning_callback = hydra.utils.instantiate(config.fine_tuning)
        callbacks.append(fine_tuning_callback)

    return callbacks


@rank_zero_only
def clear_outputs():
    current_path = pathlib.Path(".")
    for component in current_path.iterdir():
        if component.name == "checkpoints":
            shutil.rmtree(component)
        elif component.name.startswith("version_"):
            shutil.rmtree(component)
        elif component.name == "tensorboard":
            shutil.rmtree(component)
        elif component.name == "logs":
            shutil.rmtree(component)
        elif component.name == "plots":
            shutil.rmtree(component)
        elif component.name == "profile":
            shutil.rmtree(component)


def fullname(o) -> Any:
    """
    Get the full classname of an object including surrounding packages/modules/namespaces
    """
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


@contextmanager
def set_deterministic(mode, warn_only=False):
    "A contextmanager to set deterministic algorithms"

    old_mode = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    try:
        torch.use_deterministic_algorithms(mode, warn_only=warn_only)
        yield
    finally:
        torch.use_deterministic_algorithms(old_mode, warn_only=old_warn_only)
