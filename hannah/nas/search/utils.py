from copy import deepcopy
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Dict
import numpy as np

from omegaconf import OmegaConf
import yaml
from hannah.callbacks.optimization import HydraOptCallback

from hannah.nas.graph_conversion import model_to_graph
from hydra.utils import instantiate
from hannah.utils.utils import common_callbacks

msglogger = logging.getLogger(__name__)


@dataclass
class WorklistItem:
    parameters: Any
    results: Dict[str, float]
    task: Any


def save_config_to_file(current_index, configs, results):
    for num, (config, result) in enumerate(zip(configs, results)):
        nas_result_path = Path("results")
        if not nas_result_path.exists():
            nas_result_path.mkdir(parents=True, exist_ok=True)
        config_file_name = f"config_{len(current_index)+num}.yaml"
        config_path = nas_result_path / config_file_name
        with config_path.open("w") as config_file:
            config_file.write(OmegaConf.to_yaml(config))

        result_path = nas_result_path / "results.yaml"
        result_history = []
        if result_path.exists():
            with result_path.open("r") as result_file:
                result_history = yaml.safe_load(result_file)
            if not isinstance(result_history, list):
                result_history = []

        result_history.append(
                    {"config": str(config_file_name), "metrics": result}
                )

        with result_path.open("w") as result_file:
            yaml.safe_dump(result_history, result_file)


def save_graph_to_file(global_num, results: dict, model):
    nx_model = model_to_graph(model.model, model.example_feature_array.to(model.device))
    from networkx.readwrite import json_graph

    json_data = json_graph.node_link_data(nx_model)
    if not os.path.exists("../performance_data"):
        os.mkdir("../performance_data")
    with open(f"../performance_data/model_{global_num}.json", "w") as res_file:
        import json

        json.dump(
                {"graph": json_data, "metrics": results},
                res_file,
            )
        msglogger.info(f"Saved model model_{global_num}.json with results {results} to {res_file}.")

def setup_callbacks(config):
    callbacks = common_callbacks(config)
    opt_monitor = config.get("monitor", ["val_error"])
    opt_callback = HydraOptCallback(monitor=opt_monitor)
    callbacks.append(opt_callback)

    checkpoint_callback = instantiate(config.checkpoint)
    callbacks.append(checkpoint_callback)
    return callbacks, opt_monitor, opt_callback


def np_to_primitive(d):
    new_d = deepcopy(d)
    for k, v in d.items():
        if isinstance(v, (int, float)):
            new_d[k] = v
        elif isinstance(v, np.number):
            new_d[k] = v.item()
        else:
            msglogger.warning(f"No primitive registered for parameter {k} of type {type(v)}. Using original value")
    return new_d
