import argparse
import sys
import json
import os

import torch

from . import models as mod
from . import dataset
from .utils import set_seed, config_pylogger, log_execution_env_state
from .config_utils import get_config_logdir

from .config import ConfigBuilder, ConfigOption
from .callbacks.backends import OnnxTFBackend
from .callbacks.distiller import DistillerCallback

from .utils import _fullname

from .lightning_model import SpeechClassifierModule

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.memory import ModelSummary

msglogger = None


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
        backend=ConfigOption(
            default="",
            choices=["", "onnx-tf"],
            category="Backend Options",
            desc="Inference backend to use",
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


def main():
    model_name, config, config_vad, config_keyword = build_config()
    set_seed(config)
    gpu_no = config["gpu_no"]
    n_epochs = config["n_epochs"]  # max epochs
    log_dir = get_config_logdir(model_name, config)  # path for logs and checkpoints
    msglogger = config_pylogger("logging.conf", "training", log_dir)

    log_execution_env_state()

    # Configure checkpointing
    checkpoint_callback = ModelCheckpoint(
        filepath=log_dir,
        save_top_k=-1,  # with PL 0.9.0 only possible to save every epoch
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

    loggers = [
        TensorBoardLogger(log_dir + "/tb_logs", version="", name=""),
        CSVLogger(log_dir, version="", name=""),
    ]
    kwargs["logger"] = loggers

    if config["backend"] == "onnx-tf":
        backend = OnnxTFBackend()
        kwargs["callbacks"].append(backend)
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
        msglogger.info(ModelSummary(lit_module, "full"))
        lit_trainer.fit(lit_module)

        # PL TEST
        lit_trainer.test(ckpt_path=None)

        if config["profile"]:
            msglogger.info(profiler.summary())

    elif config["type"] == "eval":
        accuracy, _, _ = evaluate(model_name, config)
        print("final accuracy is", accuracy)
    elif config["type"] == "eval_vad_keyword":
        accuracy, _, _ = evaluate(model_name, config, config_vad, config_keyword)
        print("final accuracy is", accuracy)


if __name__ == "__main__":
    main()
