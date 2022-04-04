import logging
from pathlib import Path

import pandas as pd
import yaml
from genericpath import exists
from hannah_optimizer.utils import is_pareto

logger = logging.getLogger("nas_eval.extract")


def extract_models(parameters, metrics, config_metrics, extract_config):
    for task_name, task_config in extract_config.items():
        output_folder = Path("models")
        output_folder.mkdir(exist_ok=True, parents=True)
        logger.info(
            f"Extracting design points for {task_name} to {output_folder.absolute()}"
        )

        task_parameters = parameters[task_name]
        task_metrics = metrics[metrics["Task"] == task_name]

        for metric, bound in task_config.bounds.items():
            task_metrics = task_metrics[task_metrics[metric] < bound]

        selected_metric_names = [m for m in task_config.bounds.keys()]
        selected_metrics = task_metrics[selected_metric_names]

        pareto_points = is_pareto(selected_metrics.to_numpy())

        task_metrics["is_pareto"] = pareto_points
        print(task_metrics)

        candidates = task_metrics[task_metrics["is_pareto"]]

        logger.info("Network candidates:\n %s", str(candidates))

        for metric in task_config.bounds.keys():
            sorted = candidates.sort_values(metric)
            point_metrics = sorted.head(1)
            index = point_metrics["Step"]
            num = 0

            parameters = task_parameters[int(index)].flatten()
            model_parameters = parameters["model"]
            backend_parameters = {}
            if "backend" in parameters:
                backend_parameters = parameters["backend"]

            file_name = f"{task_name.lower()}_{metric}_top{num}.yaml"
            file_path = output_folder / file_name
            with file_path.open("w") as f:
                f.write("# @package _group_\n")
                f.write("\n")

                f.write("# Nas results:\n")
                for metric_name in task_config.bounds.keys():
                    metric_val = point_metrics[metric_name].item()
                    f.write(f"#   {metric_name}: {metric_val}\n")
                f.write("\n")

                if backend_parameters:
                    f.write("# Backend parameters:\n")
                    for k, v in backend_parameters.items():
                        f.write(f"#   backend.{k}={v}\n")
                    f.write("\n")
                f.write("\n")
                f.write(
                    "_target_: speech_recognition.models.factory.factory.create_cnn\n"
                )
                f.write(f"name: {task_name.lower()}_{metric}_top{num}\n")
                f.write("norm:\n")
                f.write("  target: bn\n")
                f.write("act:\n")
                f.write("  target: relu\n")

                model_parameters["qconfig"][
                    "_target_"
                ] = "speech_recognition.models.factory.qconfig.get_trax_qat_qconfig"
                model_parameters["qconfig"]["config"]["power_of_2"] = False
                model_parameters["qconfig"]["config"]["noise_prob"] = 0.7

                f.write(yaml.dump(model_parameters))
