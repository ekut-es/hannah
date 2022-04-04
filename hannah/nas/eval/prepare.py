import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

logger = logging.getLogger("nas_eval.prepare")


def prepare_summary(
    data: Dict[str, str], base_dir: str = ".", force: bool = False
) -> pd.DataFrame:
    """Prepare a summary of one or multiple nas runs

    Args:
        data (Dict[str, str]): A mapping from a short name for a nas run "e.g KWS" to a folder containing nas history file e.g. "trained_models/nas_kws/conv_net_trax"
        base_dir (str): base directory paths in data mapping are interpreted relative to base directory
        force (bool): force reconstructing of cached results ("data.pkl")
    """

    logger.info("Extracting design points")

    results_file = Path("metrics.pkl")
    parameters_file = Path("parameters.pkl")
    base_dir = Path(base_dir)
    if results_file.exists() and not force:
        changed = False
        results_mtime = results_file.stat().st_mtime
        for name, source in data.items():
            history_path = base_dir / source / "history.pkl"
            if history_path.exists():
                history_mtime = history_path.stat().st_mtime
                if history_mtime >= results_mtime:
                    changed = True
                    break
        if not changed:
            logger.info("  reading design points from saved data.pkl")
            metrics = pd.read_pickle(results_file)
            parameters = None
            with parameters_file.open("rb") as param_f:
                parameters = pickle.load(param_f)
            return metrics, parameters

    result_stack = []
    parameters_all = {}
    for name, source in data.items():
        logger.info("  Extracting design points for task: %s", name)
        history_path = base_dir / source / "history.pkl"

        if history_path.suffix == ".yml":
            with history_path.open("r") as f:
                history_file = yaml.unsafe_load(f)
        elif history_path.suffix == ".pkl":
            with history_path.open("rb") as f:
                history_file = pickle.load(f)
        else:
            raise Exception("Could not load history file: %s", str(history_path))

        results = (h.result for h in history_file)

        metrics = pd.DataFrame(results)
        metrics["Task"] = name
        metrics["Step"] = metrics.index

        parameters = [h.parameters for h in history_file]
        parameters_all[name] = parameters

        result_stack.append(metrics)

    result = pd.concat(result_stack)

    task_column = result.pop("Task")
    step_column = result.pop("Step")

    result.insert(0, "Step", step_column)
    result.insert(0, "Task", task_column)

    for column in result:
        if column not in ["Step", "Task"]:
            result[column] = result[column].astype(float)

    result.to_pickle(results_file)
    with parameters_file.open("wb") as param_f:
        pickle.dump(parameters_all, param_f)

    return result, parameters_all


def calculate_derived_metrics(
    result_metrics: pd.DataFrame, metric_definitions: Dict[str, Any]
):
    for name, metric_def in metric_definitions.items():
        logger.info("Preparing metric: %s", name)
        derived = metric_def.get("derived", None)
        if derived is not None:
            try:
                result_metrics[name] = eval(derived, {}, {"data": result_metrics})
            except Exception as e:
                logger.critical("Could not calculate derived metric %s", name)
                logger.critical(str(e))

    result_metrics = result_metrics.dropna()

    return result_metrics
