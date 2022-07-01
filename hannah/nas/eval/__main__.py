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
        base_path=hydra.utils.get_original_cwd(),
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
