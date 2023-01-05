import logging
import os
import shutil
from omegaconf import OmegaConf
import omegaconf
import torch
from hannah.callbacks.optimization import HydraOptCallback
from hannah.nas.graph_conversion import model_to_graph
from hydra.utils import instantiate
from pytorch_lightning.utilities.seed import reset_seed, seed_everything
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from hannah.utils.utils import common_callbacks
msglogger = logging.getLogger(__name__)




class ParallelModelTrainer:
    def __init__(self) -> None:
        pass

    def run_training(self, num, global_num, config):
        # num is the number of jobs global_num is the number of models to be created
        if os.path.exists(str(num)):
            shutil.rmtree(str(num))

        os.makedirs(str(num), exist_ok=True)

        try:
            os.chdir(str(num))
            config = OmegaConf.create(config)
            logger = TensorBoardLogger(".")

            seed = config.get("seed", 1234)
            if isinstance(seed, list) or isinstance(seed, omegaconf.ListConfig):
                seed = seed[0]
            seed_everything(seed, workers=True)

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

            callbacks = common_callbacks(config)
            opt_monitor = config.get("monitor", ["val_error"])
            opt_callback = HydraOptCallback(monitor=opt_monitor)
            callbacks.append(opt_callback)

            checkpoint_callback = instantiate(config.checkpoint)
            callbacks.append(checkpoint_callback)
            try:
                trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)
                model = instantiate(
                    config.module,
                    dataset=config.dataset,
                    model=config.model,
                    optimizer=config.optimizer,
                    features=config.features,
                    scheduler=config.get("scheduler", None),
                    normalizer=config.get("normalizer", None),
                    _recursive_=False,
                )
                trainer.fit(model)
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

            nx_model = model_to_graph(model.model, model.example_feature_array)
            from networkx.readwrite import json_graph

            json_data = json_graph.node_link_data(nx_model)
            if not os.path.exists("../performance_data"):
                os.mkdir("../performance_data")
            with open(f"../performance_data/model_{global_num}.json", "w") as res_file:
                import json

                json.dump(
                    {"graph": json_data, "metrics": opt_callback.result(dict=True)},
                    res_file,
                )

            return opt_callback.result(dict=True)
        finally:
            os.chdir("..")
