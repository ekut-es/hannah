import importlib
import pathlib
import shutil
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
import torch.nn as nn
import numpy as np
import logging
import os
import sys
import platform
import nvsmi
import time
import random
from pathlib import Path
from typing import Any, Callable

from git import Repo, InvalidGitRepositoryError

import hydra
from omegaconf import DictConfig

from torchvision.datasets.utils import (
    list_files,
    list_dir,
    download_and_extract_archive,
    extract_archive,
)

import pytorch_lightning
from pl_bolts.callbacks import ModuleDataMonitor, PrintTableMetricsCallback

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import GPUStatsMonitor


from .callbacks.summaries import MacSummaryCallback
from .callbacks.optimization import HydraOptCallback
from .callbacks.pruning import PruningAmountScheduler

try:
    import lsb_release  # pytype: disable=import-error

    HAVE_LSB = True
except ImportError:
    HAVE_LSB = False


class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def on_val(self):
        pass

    def on_val_end(self):
        pass

    def on_test(self):
        pass

    def on_test_end(self):
        pass

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        """ Do not use model.load """
        self.load_state_dict(
            torch.load(filename, map_location=lambda storage, loc: storage),
            strict=False,
        )


def config_pylogger(log_cfg_file, experiment_name, output_dir="logs"):
    """Configure the Python logger.
    For each execution of the application, we'd like to create a unique log directory.
    By default this directory is named using the date and time of day, so that directories
    can be sorted by recency.  You can also name your experiments and prefix the log
    directory with this name.  This can be useful when accessing experiment data from
    TensorBoard, for example.
    """
    exp_full_name = "logfile" if experiment_name is None else str(experiment_name)
    logdir = output_dir

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    log_filename = os.path.join(logdir, exp_full_name + ".log")
    if os.path.isfile(log_cfg_file):
        logging.config.fileConfig(log_cfg_file, defaults={"logfilename": log_filename})

    msglogger = logging.getLogger()
    msglogger.logdir = logdir
    msglogger.log_filename = log_filename
    msglogger.info("Log file for this run: " + os.path.realpath(log_filename))

    return msglogger


def log_execution_env_state():
    """Log information about the execution environment.
    File 'config_path' will be copied to directory 'logdir'. A common use-case
    is passing the path to a (compression) schedule YAML file. Storing a copy
    of the schedule file, with the experiment logs, is useful in order to
    reproduce experiments.
    Args:
        config_path: path to config file, used only when logdir is set
        logdir: log directory
        git_root: the path to the .git root directory
    """

    logger = logging.getLogger()

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
    if HAVE_LSB:
        logger.info("  OS: %s", lsb_release.get_lsb_information()["DESCRIPTION"])
    logger.info("  Python: %s", sys.version.replace("\n", "").replace("\r", ""))
    logger.info("  PyTorch: %s", torch.__version__)
    logger.info("  Pytorch Lightning: %s", pytorch_lightning.__version__)
    logger.info("  Numpy: %s", np.__version__)
    logger.info("  Hannah version info:")
    log_git_state(os.path.join(os.path.dirname(__file__), ".."))
    logger.info("  Command line: %s", " ".join(sys.argv))
    logger.info("  ")


def list_all_files(path, file_suffix, file_prefix=False, remove_file_beginning=""):
    subfolder = list_dir(path, prefix=True)
    files_in_folder = list_files(path, file_suffix, prefix=file_prefix)
    for subfold in subfolder:
        subfolder.extend(list_dir(subfold, prefix=True))
        if len(remove_file_beginning):
            tmp = list_files(subfold, file_suffix, prefix=False)
            tmp = [
                element
                for element in tmp
                if not element.startswith(remove_file_beginning)
            ]
            for filename in tmp:
                files_in_folder.append(os.path.join(subfold, filename))
        else:
            files_in_folder.extend(list_files(subfold, file_suffix, prefix=file_prefix))

    return files_in_folder


def extract_from_download_cache(
    filename,
    url,
    cached_files,
    target_cache,
    target_folder,
    target_test_folder="",
    clear_download=False,
    no_exist_check=False,
):
    """extracts given file from cache or donwloads first from url

        Args:
            filename (str): name of the file to download or extract
            url (str): possible url to download the file
            cached_files (list(str)): cached files in download cache
            target_cache (str): path to the folder to cache file if download necessary
            target_folder (str): path where to extract file
            target_test_folder (str, optional): folder to check if data are already there
            clear_download (bool): clear download after usage
            no_exist_check (bool): disables the check if folder exists
        """
    if len(target_test_folder) == 0:
        target_test_folder = target_folder
    if filename not in cached_files and (
        not os.path.isdir(target_test_folder) or no_exist_check
    ):
        print("download and extract: " + str(filename))
        download_and_extract_archive(
            url,
            target_cache,
            target_folder,
            filename=filename,
            remove_finished=clear_download,
        )
    elif filename in cached_files and (
        not os.path.isdir(target_test_folder) or no_exist_check
    ):
        print("extract from download_cache: " + str(filename))
        extract_archive(
            os.path.join(target_cache, filename),
            target_folder,
            remove_finished=clear_download,
        )


def auto_select_gpus(gpus=1):
    num_gpus = gpus

    gpus = list(nvsmi.get_gpus())

    gpus = list(
        sorted(gpus, key=lambda gpu: (gpu.mem_free, 1.0 - gpu.gpu_util), reverse=True)
    )

    job_num = hydra.core.hydra_config.HydraConfig.get().job.get("num", 0)

    result = []
    for i in range(num_gpus):
        num = (i + job_num) % len(gpus)
        result.append(int(gpus[num].id))

    return result


def common_callbacks(config: DictConfig):
    callbacks = []

    lr_monitor = LearningRateMonitor()
    callbacks.append(lr_monitor)

    if config.get("gpu_stats", None):
        gpu_stats = GPUStatsMonitor()
        callbacks.append(gpu_stats)

    if config.get("data_monitor", False):
        data_monitor = ModuleDataMonitor(submodules=True)
        callbacks.append(data_monitor)

    if config.get("print_metrics", False):
        metrics_printer = PrintTableMetricsCallback()
        callbacks.append(metrics_printer)

    mac_summary_callback = MacSummaryCallback()
    callbacks.append(mac_summary_callback)

    if config.get("early_stopping", None):
        stop_callback = hydra.utils.instantiate(config.early_stopping)
        callbacks.append(stop_callback)

    if config.get("pruning", None):
        pruning_scheduler = PruningAmountScheduler(
            config.pruning.amount, config.trainer.max_epochs
        )
        pruning_config = dict(config.pruning)
        del pruning_config["amount"]
        pruning_callback = hydra.utils.instantiate(
            pruning_config, amount=pruning_scheduler
        )
        callbacks.append(pruning_callback)
    return callbacks


@rank_zero_only
def clear_outputs():
    current_path = pathlib.Path(".")
    for component in current_path.iterdir():
        if component.name == "checkpoints":
            shutil.rmtree(component)
        elif component.name.startswith("version_"):
            shutil.rmtree(component)
        elif component.name == "profile":
            shutil.rmtree(component)


def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__
