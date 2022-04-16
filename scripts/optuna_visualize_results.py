from typing import Union
from pathlib import Path

import optuna
import argparse


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
