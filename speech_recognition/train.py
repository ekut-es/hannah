from collections import ChainMap, OrderedDict, defaultdict
from .config import ConfigBuilder, ConfigOption

import argparse
import os
import random
import sys
import json
import time
import math
import hashlib
import csv
import fcntl
import inspect
import importlib

from multiprocessing import cpu_count

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import itertools

from .decoder import Decoder
from . import models as mod
from . import dataset
from .utils import set_seed, config_pylogger, log_execution_env_state, EarlyStopping

# sys.path.append(os.path.join(os.path.dirname(__file__), "distiller"))
# print("__file__" + __file__)

import distiller
import distiller.model_transforms
from distiller.data_loggers import *
import distiller.apputils as apputils
import torchnet.meter as tnt
from tabulate import tabulate

from .summaries import *
from .utils import _locate, _fullname

from pytorch_lightning.trainer import Trainer
from .lightning_model import *
from .lightning_callbacks import DistillerCallback
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

msglogger = None


def get_compression(config, model, optimizer):
    if config["compress"]:
        # msglogger.info("Activating compression scheduler")
        compression_scheduler = distiller.file_config(
            model, optimizer, config["compress"]
        )
        return compression_scheduler
    else:
        return None


def get_lr_scheduler(config, optimizer):
    n_epochs = config["n_epochs"]
    lr_scheduler = config["lr_scheduler"]
    scheduler = None
    if lr_scheduler == "step":
        gamma = config["lr_gamma"]
        stepsize = config["lr_stepsize"]
        if stepsize == 0:
            stepsize = max(2, n_epochs // 15)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == "multistep":
        gamma = config["lr_gamma"]
        steps = config["lr_steps"]
        if steps == [0]:
            steps = itertools.count(max(1, n_epochs // 10), max(1, n_epochs // 10))

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=gamma)

    elif lr_scheduler == "exponential":
        gamma = config["lr_gamma"]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    elif lr_scheduler == "plateau":
        gamma = config["lr_gamma"]
        patience = config["lr_patience"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=gamma,
            patience=patience,
            threshold=0.00000001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
        )

    else:
        raise Exception("Unknown learing rate scheduler: {}".format(lr_scheduler))

    return scheduler


def get_optimizer(config, model):

    if config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["lr"],
            nesterov=config["use_nesterov"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
        )
    elif config["optimizer"] == "adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(),
            lr=config["lr"],
            rho=config["opt_rho"],
            eps=config["opt_eps"],
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer"] == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=config["lr"],
            lr_decay=config["lr_decay"],
            weight_decay=config["weight_decay"],
        )

    elif config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            betas=config["opt_betas"],
            eps=config["opt_eps"],
            weight_decay=config["weight_decay"],
            amsgrad=config["use_amsgrad"],
        )
    elif config["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=config["lr"],
            alpha=config["opt_alpha"],
            eps=config["opt_eps"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
        )
    else:
        raise Exception("Unknown Optimizer: {}".format(config["optimizer"]))

    return optimizer


def get_loss_function(model, config):

    ce = nn.CrossEntropyLoss()

    def ce_loss_func(scores, labels):
        scores = scores.view(scores.size(0), -1)
        return ce(scores, labels)

    criterion = ce_loss_func

    try:
        criterion = model.get_loss_function()
    except Exception as e:
        print(str(e))
        if "loss" in config:
            if config["loss"] == "cross_entropy":
                criterion = nn.CrossEntropyLoss()
            elif config["loss"] == "ctc":
                criterion = ce_loss_func
            else:
                raise Exception(
                    "Loss function not supported: {}".format(config["loss"])
                )

    return criterion


def get_output_dir(model_name, config):

    output_dir = os.path.join(config["output_dir"], config["experiment_id"], model_name)

    if config["compress"]:
        compressed_name = config["compress"]
        compressed_name = os.path.splitext(os.path.basename(compressed_name))[0]
        output_dir = os.path.join(output_dir, compressed_name)

    output_dir = os.path.abspath(output_dir)

    return output_dir


def get_config_logdir(model_name, config):
    return os.path.join(
        get_output_dir(model_name, config), "configs", config["config_hash"]
    )


def get_model(config, config2=None, model=None, vad_keyword=0):
    if not model:
        if vad_keyword == 0:
            model = _locate(config["model_class"])(config)
            if config["input_file"]:
                model.load(config["input_file"])
        elif vad_keyword == 1:
            model = _locate(config2["model_class"])(config2)
            if config["input_file_vad"]:
                model.load(config["input_file_vad"])
        else:
            model = _locate(config2["model_class"])(config2)
            if config["input_file_keyword"]:
                model.load(config["input_file_keyword"])
    return model


def reset_symlink(src, dest):
    if os.path.exists(dest):
        os.unlink(dest)
    os.symlink(src, dest)


def dump_config(output_dir, config):
    """Dumps the configuration to json format

    Creates file config.json in output_dir

    Parameters
    ----------
    output_dir : str
       Output directory
    config  : dict
       Configuration to dump
    """

    with open(os.path.join(output_dir, "config.json"), "w") as o:
        s = json.dumps(dict(config), default=lambda x: str(x), indent=4, sort_keys=True)
        o.write(s)


def save_model(
    output_dir, model, test_set=None, config=None, model_prefix="", msglogger=None
):
    """Creates serialization of the model for later inference, evaluation

    Creates the following files:

    - model.pt: Serialized version of network parameters in pytorch
    - model.json: Serialized version of network parameters in json format
    - model.onnx: full model including paramters in onnx format

    Parameters
    ----------

    output_dir : str
        Directory to put serialized models
    model : torch.nn.Module
        Model to serialize
    test_set : dataset.SpeechDataset
        DataSet used to derive dummy input to use for onnx export.
        If None no onnx will be generated
    """

    # TODO model save doesnt work "AttributeError: model has no attribute save"
    # msglogger.info("saving best model...")
    # model.save(os.path.join(output_dir, model_prefix+"model.pt"))

    msglogger.info("saving weights to json...")
    filename = os.path.join(output_dir, model_prefix + "model.json")
    state_dict = model.state_dict()
    with open(filename, "w") as f:
        json.dump(state_dict, f, default=lambda x: x.tolist(), indent=2)

    msglogger.info("saving onnx...")
    try:
        dummy_width, dummy_height = test_set.width, test_set.height
        dummy_input = torch.randn((1, dummy_height, dummy_width))

        if config["cuda"]:
            dummy_input = dummy_input.cuda()

        torch.onnx.export(
            model,
            dummy_input,
            os.path.join(output_dir, model_prefix + "model.onnx"),
            verbose=False,
        )
    except Exception as e:
        msglogger.error("Could not export onnx model ...\n {}".format(str(e)))


def build_config(extra_config={}):
    output_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "trained_models"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[x.value for x in list(mod.ConfigType)],
        default="ekut-raw-cnn3",
        type=str,
    )
    parser.add_argument("--config", default="", type=str)
    parser.add_argument("--config_vad", default="", type=str)
    parser.add_argument("--config_keyword", default="", type=str)
    parser.add_argument(
        "--dataset",
        choices=["keywords", "hotword", "vad", "keywords_and_noise"],
        default="keywords",
        type=str,
    )
    config, _ = parser.parse_known_args()

    model_name = config.model
    dataset_name = config.dataset

    default_config = {}
    if config.config:
        with open(config.config, "r") as f:
            default_config = json.load(f)
            model_name = default_config["model_name"]

            # Delete model from config for now to avoid showing
            # them as commandline otions
            if "model_name" in default_config:
                del default_config["model_name"]
            else:
                print("Your model config does not include a model_name")
                print(" these configurations are not loadable")
                sys.exit(-1)
            if "model_class" in default_config:
                del default_config["model_class"]
            if "type" in default_config:
                del default_config["type"]
            if "dataset" in default_config:
                dataset_name = default_config["dataset"]
                del default_config["dataset"]
            if "dataset_cls" in default_config:
                del default_config["dataset_cls"]
            if "config" in default_config:
                del default_config["config"]
            if "config_vad" in default_config:
                del default_config["config_vad"]
            if "config_keyword" in default_config:
                del default_config["config_keyword"]
            if "model" in default_config:
                del default_config["model"]

    default_config_vad = {}
    if config.config_vad:
        with open(config.config_vad, "r") as f:
            default_config_vad = json.load(f)
            model_name = default_config_vad["model_name"]

            # Delete model from config for now to avoid showing
            # them as commandline otions
            if "model_name" in default_config_vad:
                del default_config_vad["model_name"]
            else:
                print("Your model config does not include a model_name")
                print(" these configurations are not loadable")
                sys.exit(-1)
            if "model_class" in default_config_vad:
                del default_config_vad["model_class"]
            if "type" in default_config:
                del default_config_vad["type"]

    default_config_keyword = {}
    if config.config_keyword:
        with open(config.config_keyword, "r") as f:
            default_config_keyword = json.load(f)
            model_name = default_config_keyword["model_name"]

            # Delete model from config for now to avoid showing
            # them as commandline otions
            if "model_name" in default_config_keyword:
                del default_config_keyword["model_name"]
            else:
                print("Your model config does not include a model_name")
                print(" these configurations are not loadable")
                sys.exit(-1)
            if "model_class" in default_config_keyword:
                del default_config_keyword["model_class"]
            if "type" in default_config_keyword:
                del default_config_keyword["type"]

    global_config = dict(
        cuda=ConfigOption(
            default=torch.cuda.is_available(), desc="Enable / disable cuda"
        ),
        n_epochs=ConfigOption(default=500, desc="Number of epochs for training"),
        profile=ConfigOption(default=False, desc="Enable profiling"),
        dump_test=ConfigOption(
            default=False, desc="Dump test set to <output_directory>/test_data"
        ),
        num_workers=ConfigOption(
            desc="Number of worker processes used for data loading (using a number > 0) makes results non reproducible",
            default=0,
        ),
        fold_bn=ConfigOption(
            default=-1, desc="Do BatchNorm folding at freeze at the given epoch"
        ),
        optimizer=ConfigOption(
            default="sgd",
            desc="Optimizer to choose",
            category="Optimizer Config",
            choices=["sgd", "adadelta", "adagrad", "adam", "rmsprop"],
        ),
        opt_rho=ConfigOption(
            category="Optimizer Config",
            desc="Parameter rho for Adadelta optimizer",
            default=0.9,
        ),
        opt_eps=ConfigOption(
            category="Optimizer Config",
            desc="Paramter eps for Adadelta and Adam and SGD",
            default=1e-06,
        ),
        opt_alpha=ConfigOption(
            category="Optimizer Config",
            desc="Parameter alpha for RMSprop",
            default=0.99,
        ),
        lr_decay=ConfigOption(
            category="Optimizer Config",
            desc="Parameter lr_decay for optimizers",
            default=0,
        ),
        use_amsgrad=ConfigOption(
            category="Optimizer Config",
            desc="Use amsgrad with Adam optimzer",
            default=False,
        ),
        opt_betas=ConfigOption(
            category="Optimizer Config",
            desc="Parameter betas for Adam optimizer",
            default=[0.9, 0.999],
        ),
        momentum=ConfigOption(
            category="Optimizer Config", desc="Momentum for SGD optimizer", default=0.9
        ),
        weight_decay=ConfigOption(
            category="Optimizer Config",
            desc="Weight decay for optimizer",
            default=0.00001,
        ),
        use_nesterov=ConfigOption(
            category="Optimizer Config",
            desc="Use nesterov momentum with SGD optimizer",
            default=False,
        ),
        auto_lr=ConfigOption(
            category="Learning Rate Config",
            default=False,
            desc="Determines the learning rate automatically",
        ),
        lr=ConfigOption(
            category="Learning Rate Config", desc="Initial Learining Rate", default=0.1
        ),
        lr_scheduler=ConfigOption(
            category="Learning Rate Config",
            desc="Learning Rate Scheduler to use",
            choices=["step", "multistep", "exponential", "plateau"],
            default="step",
        ),
        lr_gamma=ConfigOption(
            category="Learning Rate Config",
            desc="Parameter gamma for lr scheduler",
            default=0.75,
        ),
        lr_stepsize=ConfigOption(
            category="Learning Rate Config",
            desc="Stepsize for step scheduler",
            default=0,
        ),
        lr_steps=ConfigOption(
            category="Learning Rate Config",
            desc="List of steps for multistep scheduler",
            default=[0],
        ),
        lr_patience=ConfigOption(
            category="Learning Rate Config",
            desc="Parameter patience for plateau scheduler",
            default=10,
        ),
        early_stopping=ConfigOption(
            default=0,
            desc="Stops the training if the validation loss has not improved for the last EARLY_STOPPING epochs",
        ),
        limits_datasets=ConfigOption(
            default=[1.0, 1.0, 1.0],
            desc="One value for train, validation and test dataset. Decimal number for percentage of dataset. Natural number for exact sample count.",
        ),
        fast_dev_run=ConfigOption(
            default=False,
            desc="Runs 1 batch of train, test and val to find any bugs (ie: a sort of unit test).",
        ),
        batch_size=ConfigOption(
            default=128, desc="Default minibatch size for training"
        ),
        seed=ConfigOption(default=0, desc="Seed for Random number generators"),
        input_file=ConfigOption(
            default="",
            desc="Input model file for finetuning (.pth) or code generation (.onnx)",
        ),
        input_file_vad=ConfigOption(
            default="",
            desc="Input vad model file for combined evaluation of vad and keyword spotting",
        ),
        input_file_keyword=ConfigOption(
            default="",
            desc="Input keyword model file for combined evaluation of vad and keyword spotting",
        ),
        output_dir=ConfigOption(
            default=output_dir,
            desc="Toplevel directory for output of trained models and logs",
        ),
        gpu_no=ConfigOption(default=0, desc="Number of GPU to use for training"),
        compress=ConfigOption(
            default="", desc="YAML config file for nervana distiller"
        ),
        tblogger=ConfigOption(
            default=False,
            desc="Enable logging of learning progress and network parameter statistics to Tensorboard",
        ),
        experiment_id=ConfigOption(
            default="test",
            desc="Unique id to identify the experiment, overwrites all output files with same experiment id, output_dir, and model_name",
        ),
    )

    mod_cls = mod.find_model(model_name)
    dataset_cls = dataset.find_dataset(dataset_name)
    builder = ConfigBuilder(
        default_config,
        mod.find_config(model_name),
        dataset_cls.default_config(),
        global_config,
        extra_config,
    )

    parser = builder.build_argparse(parser)

    parser.add_argument(
        "--type",
        choices=["train", "eval", "eval_vad_keyword"],
        default="train",
        type=str,
    )
    config = builder.config_from_argparse(parser)

    config["model_class"] = _fullname(mod_cls)
    default_config_vad["model_class"] = mod.find_model(
        "small-vad"
    )  # als command line option umändern
    default_config_keyword["model_class"] = mod.find_model("honk-res15")
    config["model_name"] = model_name
    config["dataset"] = dataset_name
    config["dataset_cls"] = _fullname(dataset_cls)

    return (model_name, config, default_config_vad, default_config_keyword)


from pytorch_lightning.core.lightning import ModelSummary


def main():
    model_name, config, config_vad, config_keyword = build_config()
    set_seed(config)
    # Set deterministic mode for CUDNN backend
    # Check if the performance penalty might be too high

    gpu_no = config["gpu_no"]
    n_epochs = config["n_epochs"]  # max epochs
    log_dir = get_config_logdir(model_name, config)  # path for logs and checkpoints
    msglogger = config_pylogger("logging.conf", "lightning-logger", log_dir)

    # Configure checkpointing
    checkpoint_callback = ModelCheckpoint(
        filepath=log_dir,
        save_top_k=-1,  #  with PL 0.9.0 only possible to save every epoch
        verbose=True,
        monitor="checkpoint_on",
        mode="min",
        prefix="",
    )

    lit_module = SpeechClassifierModule(
        dict(config), log_dir, msglogger
    )  # passing logdir for custom json save after training omit double fnccall

    kwargs = {
        "max_epochs": n_epochs,
        "default_root_dir": log_dir,
        "row_log_interval": 1,  # enables logging of metrics per step/batch
        "checkpoint_callback": checkpoint_callback,
        "callbacks": [],
    }

    # TODO distiller only available without auto_lr because compatibility issues
    if config["compress"] and not config["auto_lr"]:
        callbacks = kwargs["callbacks"]
        callbacks.append(
            DistillerCallback(config["compress"], fold_bn=config["fold_bn"])
        )
        kwargs.update({"callbacks": callbacks})

    if config["cuda"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        kwargs.update({"gpus": [gpu_no]})

    if "limits_datasets" in config:
        limits = config["limits_datasets"]
        kwargs.update(
            {
                "limit_train_batches": limits[0],
                "limit_val_batches": limits[1],
                "limit_test_batches": limits[2],
            }
        )

    if config["fast_dev_run"]:
        kwargs.update({"fast_dev_run": True})

    if config["type"] == "train":

        if config["profile"]:
            profiler = AdvancedProfiler()
            kwargs.update({"profiler": profiler})

        # INIT PYTORCH-LIGHTNING
        lit_trainer = Trainer(**kwargs)

        if config["auto_lr"]:
            # run lr finder (counts as one epoch)
            lr_finder = lit_trainer.lr_find(lit_module)
            # inspect results
            fig = lr_finder.plot()
            fig.savefig(f"{log_dir}/learing_rate.png")
            # recreate module with updated config
            suggested_lr = lr_finder.suggestion()
            config["lr"] = suggested_lr
            lit_module = SpeechClassifierModule(dict(config), log_dir, msglogger)

        # PL TRAIN
        print(ModelSummary(lit_module, "full"))
        lit_trainer.fit(lit_module)

        # PL TEST
        lit_trainer.test(ckpt_path=None)

        if config["profile"]:
            print(profiler.summary())

    elif config["type"] == "eval":
        accuracy, _, _ = evaluate(model_name, config)
        print("final accuracy is", accuracy)
    elif config["type"] == "eval_vad_keyword":
        accuracy, _, _ = evaluate(model_name, config, config_vad, config_keyword)
        print("final accuracy is", accuracy)


if __name__ == "__main__":
    main()
