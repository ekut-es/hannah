#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
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
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import hannah.conf  # noqa
from hannah.nas.graph_conversion import model_to_graph
from hannah.nas.parametrization import SearchSpace
from hannah.nas.performance_prediction.simple import to_dgl_graph


@hydra.main(config_name="characterize", config_path="../conf", version_base="1.2")
def main(config: DictConfig):
    """

    Args:
      config: DictConfig:
      config: DictConfig:
      config: DictConfig:

    Returns:

    """
    random_state = np.random.RandomState(seed=1234)
    search_space = SearchSpace(config.nas.parametrization, random_state)

    backend = instantiate(config.backend)

    for i in range(0, 20000):
        parameters = search_space.get_random()
        config = OmegaConf.merge(config, parameters.flatten())

        skip = False
        for board in config.backend.board:
            result_path = Path(board) / f"config_{i}.yaml"
            if result_path.exists():
                skip = True
        if skip:
            logging.info("Skipping config_%d", i)
            continue

        logging.info(OmegaConf.to_yaml(config.model))
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
        model.setup("test")
        model.eval()

        network_graph = model_to_graph(model.model, model.example_feature_array)
        results = backend.characterize(model)

        for result in results:
            board = result["board"]
            directory = Path(board)

            directory.mkdir(exist_ok=True)

            config_file_name = f"config_{i}.yaml"
            config_path = directory / config_file_name
            with config_path.open("w") as config_file:
                config_file.write(OmegaConf.to_yaml(config))

            result_path = directory / "results.yaml"
            result_history = []
            if result_path.exists():
                with result_path.open("r") as result_file:
                    result_history = yaml.safe_load(result_file)
                if not isinstance(result_history, list):
                    result_history = []

            result_history.append({"config": str(config_file_name), "metrics": result})

            with result_path.open("w") as result_file:
                yaml.safe_dump(result_history, result_file)


if __name__ == "__main__":
    main()
