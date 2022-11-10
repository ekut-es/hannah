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
