import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import seaborn as sns
import yaml

logger = logging.getLogger("nas_eval.plot")


def _rename_metric(name, name_dict):
    if name in name_dict:
        return name_dict[name]
    return name


def plot_comparison(
    data: pd.DataFrame,
    metrics: List[str],
    name: str,
    metric_config: Optional[Dict[str, Any]] = None,
):
    sns.set_theme(style="white")

    y = metrics[0]
    x = metrics[1]
    size = metrics[2] if len(metrics) >= 3 else None
    style = metrics[3] if len(metrics) >= 4 else "Task"
    hue = "Task"

    # Filter:
    # result[result['Accuracy [$\%$]'] > 85.0]

    name_dict = {}
    for index, config in metric_config.items():
        if "name" in config:
            name_dict[index] = config.name

    data = data.rename(columns=name_dict)
    x, y, size, style, hue = [
        _rename_metric(metric, name_dict) for metric in [x, y, size, style, hue]
    ]

    # Filter:
    data = data[data["Accuracy [%]"] > 75.0]

    plot = sns.relplot(
        x=x,
        y=y,
        size=size,
        style=style,
        hue=hue,
        sizes=(40, 400),
        alpha=0.7,
        palette="muted",
        height=6,
        data=data,
    )

    logger.info("  Saving comparison plot in %s", os.getcwd())

    name = f"comparison_{name}"
    for ext in [".png", ".pdf"]:
        ext_name = name + ext
        plot.savefig(ext_name)
        logger.info("    - %s", ext_name)
