import importlib
import torch
import torch.nn as nn
import numpy as np
import logging
import os
import sys
import platform
from pathlib import Path
from typing import Any, Callable

from git import Repo, InvalidGitRepositoryError

from torchvision.datasets.utils import list_files, list_dir

try:
    import lsb_release

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


def log_execution_env_state(distiller_gitroot="."):
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
    logger.info("  Numpy: %s", np.__version__)
    logger.info("  Distiller Info:")
    log_git_state(distiller_gitroot)
    logger.info("  Speech Recognition info:")
    log_git_state(os.path.join(os.path.dirname(__file__), ".."))
    logger.info("  Command line: %s", " ".join(sys.argv))
    logger.info("  ")


def load_module(path):
    """small utility to automatically load modules from path"""
    path = Path(path)
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_all_files(path, file_suffix, file_prefix=False):
    subfolder = list_dir(path, prefix=True)
    files_in_folder = list_files(path, file_suffix, prefix=file_prefix)
    for element in subfolder:
        subfolder.extend(list_dir(element, prefix=True))
        files_in_folder.extend(list_files(element, file_suffix, prefix=file_prefix))

    return files_in_folder
