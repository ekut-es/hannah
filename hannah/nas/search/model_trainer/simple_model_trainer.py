from copy import deepcopy
import logging
import os
import shutil
from omegaconf import DictConfig, OmegaConf
import omegaconf
import torch
import torch.nn as nn
from hannah.callbacks.optimization import HydraOptCallback
from hannah.nas.graph_conversion import model_to_graph
from hydra.utils import instantiate
from pytorch_lightning.utilities.seed import reset_seed, seed_everything
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from hannah.nas.search.utils import save_graph_to_file, setup_callbacks

from hannah.utils.utils import common_callbacks
msglogger = logging.getLogger(__name__)


class SimpleModelTrainer:
    def __init__(self) -> None:
        pass

    def build_model(self, model, parameters):

        model_instance = deepcopy(model)

        for k, p in model_instance.parametrization(flatten=True).items():
            p.set_current(parameters[k])

        model_instance.initialize()
        model = model_instance

        return model

    def run_training(self, model, num, global_num, config):
        # num is the number of jobs global_num is the number of models to be created
        if os.path.exists(str(num)):
            shutil.rmtree(str(num))

        os.makedirs(str(num), exist_ok=True)

        try:
            os.chdir(str(num))
            config = OmegaConf.create(config)
            logger = TensorBoardLogger(".")

            self.setup_seed(config)
            self.setup_gpus(num, config, logger)
            callbacks, opt_monitor, opt_callback = setup_callbacks(config)
            try:
                trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)
                module = model
                trainer.fit(module)
                ckpt_path = "best"
                if trainer.fast_dev_run:
                    logging.warning(
                        "Trainer is in fast dev run mode, switching off loading of best model for test"
                    )
                    ckpt_path = None

                reset_seed()
                trainer.validate(ckpt_path=ckpt_path, verbose=False)
            except Exception as e:
                msglogger.critical("Training failed with exception")
                msglogger.critical(str(e))
                res = {}
                for monitor in opt_monitor:
                    res[monitor] = float("inf")

            save_graph_to_file(global_num, opt_callback, module)

            return opt_callback.result(dict=True)
        finally:
            os.chdir("..")

    def set_result_handler(self, result_handler):
        self.result_handler = result_handler

    def setup_seed(self, config):
        seed = config.get("seed", 1234)
        if isinstance(seed, list) or isinstance(seed, omegaconf.ListConfig):
            seed = seed[0]
        seed_everything(seed, workers=True)

    def setup_gpus(self, num, config, logger):
        if config.trainer.gpus is not None:
            if isinstance(config.trainer.gpus, int):
                num_gpus = config.trainer.gpus
                gpu = num % num_gpus
            elif len(config.trainer.gpus) == 0:
                num_gpus = torch.cuda.device_count()
                gpu = num % num_gpus
            else:
                gpu = config.trainer.gpus[num % len(config.trainer.gpus)]

            if gpu >= torch.cuda.device_count():
                logger.warning(
                        "GPU %d is not available on this device using GPU %d instead",
                        gpu,
                        gpu % torch.cuda.device_count(),
                    )
                gpu = gpu % torch.cuda.device_count()

            config.trainer.gpus = [gpu]
