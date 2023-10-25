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
import io
import logging
import os
from unittest import result

import hydra
from omegaconf import OmegaConf

from .extract import extract_models
from .plot import plot_comparison
from .prepare import calculate_derived_metrics, prepare_summary

logger = logging.getLogger("nas_eval")


@hydra.main(config_path="../../conf/nas", config_name="eval", version_base="1.2")
def main(config):
    logger.info("Current working directory %s", os.getcwd())
    result_metrics, parameters = prepare_summary(
        config.data,
        base_dir=hydra.utils.get_original_cwd(),
        force=config.get("force", False),
    )

    metrics_info = io.StringIO()
    result_metrics.info(buf=metrics_info)

    logger.info("Statistics of result data:\n %s", metrics_info.getvalue())

    logger.info("Raw result metrics(head)\n %s", str(result_metrics.head()))

    derived_metrics = calculate_derived_metrics(result_metrics, config.metrics)
    logger.info("Derived result metrics(head)\n %s", str(derived_metrics.head()))

    logger.info("Plotting results")

    for plot in config.plots:
        if plot.type == "comparison":
            logger.info("  plotting %s of type %s", plot.name, plot.type)
            plot_comparison(
                derived_metrics, plot.metrics, plot.name, metric_config=config.metrics
            )
        else:
            logger.warning("unknown plot type: %s")

    extract_models(parameters, derived_metrics, config.metrics, config.extract)


if __name__ == "__main__":
    main()
