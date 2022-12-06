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

import hydra
from omegaconf import DictConfig

from .. import conf  # noqa
from ..logo import print_logo


@hydra.main(config_name="config", config_path="../conf", version_base="1.2")
def main(config: DictConfig):

    print_logo()
    # Lazily Imported to get quicker tab completion
    from ..train import handle_dataset, nas, train
    from ..utils.utils import log_execution_env_state

    logging.captureWarnings(True)
    try:
        log_execution_env_state()
        if config.get("dataset_creation", None) is not None:
            handle_dataset(config)
        if config.get("nas", None) is not None:
            return nas(config)
        else:
            return train(config)
    except Exception as e:
        logging.exception("Exception Message: %s", str(e))
        raise e


if __name__ == "__main__":
    main()
