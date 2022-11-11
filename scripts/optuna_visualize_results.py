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
import argparse
from pathlib import Path
from typing import Union

import optuna


def get_storage(
    storage: Union[str, optuna.storages.BaseStorage]
) -> optuna.storages.BaseStorage:
    if isinstance(storage, str):
        if storage.startswith("redis"):
            return optuna.storages.RedisStorage(storage)
        else:
            return optuna.storages.RDBStorage(storage)
    return


def main(args):
    storage = get_storage(args.storage)

    study_names = args.study_name

    if not study_names or args.list:
        study_names = [s.study_name for s in storage.get_all_study_summaries()]

        if args.list:
            print("Available studies")
            for study_name in study_names:
                print(" -", study_name)
            return

    studies = []
    for study_name in study_names:
        study = optuna.load_study(study_name, storage)
        studies.append(study)

    out_base = Path(args.out)

    for study in studies:
        # Plot History
        out = out_base / study.study_name
        out.mkdir(parents=True, exist_ok=True)

        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(out / "history.png")
        fig.write_image(out / "history.pdf")

        # Plot Parameter Importance
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(out / "importance.png")
        fig.write_image(out / "importance.pdf")

        # Plot Parameter Importance
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(out / "parallel_coordinate.png")
        fig.write_image(out / "parallel_coordinate.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Show optuna hyperopt results")
    parser.add_argument(
        "storage",
        metavar="STORAGE",
        help="The optuna storage url e.g.:  sqlite:///trial.sqlite",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--study-name",
        nargs="+",
        type=list,
        default=[],
        help="studies to plot, by default plots all studies",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available studies and exit"
    )
    parser.add_argument("-o", "--out", type=str, default="out", help="Output folder")

    args = parser.parse_args()

    main(args)
