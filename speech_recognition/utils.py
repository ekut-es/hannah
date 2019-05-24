import torch
import torch.nn as nn
import numpy as np
import random
import logging
import os
import json

import sys
import platform
from git import Repo, InvalidGitRepositoryError
try:
    import lsb_release
    HAVE_LSB = True
except ImportError:
    HAVE_LSB = False

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop
        

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if config["cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)


def config_pylogger(log_cfg_file, experiment_name, output_dir='logs'):
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

    log_filename = os.path.join(logdir, exp_full_name + '.log')
    if os.path.isfile(log_cfg_file):
        logging.config.fileConfig(log_cfg_file, defaults={'logfilename': log_filename})

    
    msglogger = logging.getLogger()
    msglogger.logdir = logdir
    msglogger.log_filename = log_filename
    msglogger.info('Log file for this run: ' + os.path.realpath(log_filename))

    return msglogger


def log_execution_env_state(distiller_gitroot='.'):
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
            logger.info("    Cannot find a Git repository.  You probably downloaded an archive")
            return

        if repo.is_dirty():
            logger.info("    Git is dirty")
        try:
            branch_name = repo.active_branch.name
        except TypeError:
            branch_name = "None, Git is in 'detached HEAD' state"
        logger.info("    Active Git branch: %s", branch_name)
        logger.info("    Git commit: %s" % repo.head.commit.hexsha)

    logger.info("  Number of CPUs: %d", len(os.sched_getaffinity(0)))
    logger.info("  Number of GPUs: %d", torch.cuda.device_count())
    logger.info("  CUDA version: %s", torch.version.cuda)
    logger.info("  CUDNN version: %s", torch.backends.cudnn.version())
    logger.info("  Kernel: %s", platform.release())
    if HAVE_LSB:
        logger.info("  OS: %s", lsb_release.get_lsb_information()['DESCRIPTION'])
    logger.info("  Python: %s", sys.version.replace("\n", "").replace("\r", ""))
    logger.info("  PyTorch: %s", torch.__version__)
    logger.info("  Numpy: %s", np.__version__)
    logger.info("  Distiller Info:")
    log_git_state(distiller_gitroot)
    logger.info("  Speech Recognition info:")
    log_git_state(os.path.join(os.path.dirname(__file__), '..'))
    logger.info("  Command line: %s", " ".join(sys.argv))
    logger.info("  ")
