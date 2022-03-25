import logging
import os
from pathlib import Path
import numpy as np
import shutil
from collections import defaultdict
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.utilities.seed import reset_seed, seed_everything
from pytorch_lightning.utilities.distributed import rank_zero_info, rank_zero_only

from hydra.utils import instantiate
from omegaconf import DictConfig

from . import conf  # noqa
from .callbacks.summaries import MacSummaryCallback
from .callbacks.optimization import HydraOptCallback
from .callbacks.pruning import PruningAmountScheduler
from .utils import (
    log_execution_env_state,
    auto_select_gpus,
    common_callbacks,
    clear_outputs,
)


@rank_zero_only
def handleDataset(config=DictConfig):
    lit_module = instantiate(
        config.module,
        dataset=config.dataset,
        model=config.model,
        optimizer=config.optimizer,
        features=config.features,
        scheduler=config.get("scheduler", None),
        normalizer=config.get("normalizer", None),
        _recursive_=False,
    )
    lit_module.prepare_data()


def train(config: DictConfig):
    test_output = []
    results = []
    if isinstance(config.seed, int):
        config.seed = [config.seed]

    for seed in config.seed:
        seed_everything(seed, workers=True)
        if not torch.cuda.is_available():
            config.trainer.gpus = None

        if isinstance(config.trainer.gpus, int):
            config.trainer.gpus = auto_select_gpus(config.trainer.gpus)

        if not config.trainer.fast_dev_run and not config.get("resume", False):
            clear_outputs()

        logging.info("Configuration: ")
        logging.info(OmegaConf.to_yaml(config))
        logging.info("Current working directory %s", os.getcwd())
        lit_module = instantiate(
            config.module,
            dataset=config.dataset,
            model=config.model,
            optimizer=config.optimizer,
            features=config.features,
            scheduler=config.get("scheduler", None),
            normalizer=config.get("normalizer", None),
            gpus=config.trainer.get("gpus", None),
            _recursive_=False,
        )

        profiler = None
        if config.get("profiler", None):
            profiler = instantiate(config.profiler)

        logger = [
            TensorBoardLogger(".", version=None, name="", default_hp_metric=False)
        ]
        if config.trainer.get("stochastic_weight_avg", False):
            logging.critical(
                "CSVLogger is not compatible with logging with SWA, disabling csv logger"
            )
        else:
            logger.append(CSVLogger(".", version=None, name=""))

        callbacks = []
        if config.get("backend", None):
            backend = instantiate(config.backend)
            callbacks.append(backend)

        callbacks.extend(list(common_callbacks(config)))

        opt_monitor = config.get("monitor", ["val_error"])
        opt_callback = HydraOptCallback(monitor=opt_monitor)
        callbacks.append(opt_callback)

        checkpoint_callback = instantiate(config.checkpoint)
        callbacks.append(checkpoint_callback)

        # INIT PYTORCH-LIGHTNING
        lit_trainer = instantiate(
            config.trainer,
            profiler=profiler,
            callbacks=callbacks,
            logger=logger,
            _convert_="partial",
        )

        if config["auto_lr"]:
            # run lr finder (counts as one epoch)
            lr_finder = lit_trainer.lr_find(lit_module)

            # inspect results
            fig = lr_finder.plot()
            fig.savefig("./learning_rate.png")

            # recreate module with updated config
            suggested_lr = lr_finder.suggestion()
            config["lr"] = suggested_lr

        lit_trainer.tune(lit_module)

        logging.info("Starting training")
        # PL TRAIN
        ckpt_path = None
        if config.get("resume", False):
            expected_ckpt_path = Path(".") / "checkpoints" / "last.ckpt"
            # breakpoint()
            if expected_ckpt_path.exists():
                logging.info(
                    "Resuming training from checkpoint: %s", str(expected_ckpt_path)
                )
                ckpt_path = str(expected_ckpt_path)
            else:
                logging.info(
                    "Checkpoint '%s' not found restarting training from scratch",
                    str(expected_ckpt_path),
                )
        lit_trainer.fit(lit_module, ckpt_path=ckpt_path)
        ckpt_path = "best"

        if not lit_trainer.fast_dev_run:
            reset_seed()
            lit_trainer.validate(ckpt_path=ckpt_path, verbose=False)

            # PL TEST
            reset_seed()
            lit_trainer.test(ckpt_path=ckpt_path, verbose=False)

            lit_module.save()
            if checkpoint_callback and checkpoint_callback.best_model_path:
                shutil.copy(checkpoint_callback.best_model_path, "best.ckpt")

        test_output.append(opt_callback.test_result())
        results.append(opt_callback.result())

    test_sum = defaultdict(int)
    for output in test_output:
        for k, v in output.items():
            if v.numel() == 1:
                test_sum[k] += v.item()
            else:
                test_sum[k] += v

    rank_zero_info("Averaged Test Metrics:")

    for k, v in test_sum.items():
        rank_zero_info(k + " : " + str(v / len(test_output)))
    rank_zero_info("validation_error : " + str(np.sum(results) / len(results)))

    if len(results) == 1:
        return results[0]
    else:
        return results


def nas(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    nas_trainer = instantiate(config.nas, parent_config=config, _recursive_=False)
    nas_trainer.run()


@hydra.main(config_name="config", config_path="conf")
def main(config: DictConfig):
    logging.captureWarnings(True)
    try:
        log_execution_env_state()
        if config.get("dataset_creation", None) is not None:
            handleDataset(config)
        if config.get("nas", None) is not None:
            return nas(config)
        else:
            return train(config)
    except Exception as e:
        logging.exception("Exception Message: %s", str(e))
        raise e


if __name__ == "__main__":
    main()
