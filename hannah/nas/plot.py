from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tikzplotlib

import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot

from .utils import is_pareto


def plot_multi(data, cols=None, offset=60, **kwargs):

    plt.rcParams["figure.figsize"] = (10, 3)
    from pandas.plotting._matplotlib.style import get_standard_colors

    # Get default color style from pandas - can be changed to any other color list
    if cols is None:
        cols = data.columns
    if len(cols) == 0:
        return

    colors = get_standard_colors(num_colors=len(cols))

    host = host_subplot(111, axes_class=AA.Axes)

    # First axis
    host.plot(data.loc[:, cols[0]].to_numpy(), label=cols[0], color=colors[0], **kwargs)
    host.set_ylabel(cols[0])

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = host.twinx()
        new_fixed_axis = ax_new.get_grid_helper().new_fixed_axis
        ax_new.axis["right"] = new_fixed_axis(
            loc="right", axes=ax_new, offset=(offset * (n - 1), 0)
        )
        ax_new.axis["right"].toggle(all=True)
        ax_new.plot(
            data.loc[:, cols[n]].to_numpy(), label=cols[n], color=colors[n], **kwargs
        )
        ax_new.set_ylabel(ylabel=cols[n])

    host.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncol=3,
        mode="expand",
        borderaxespad=0.0,
    )
    return host


def plot_pareto_front(history, output_folder="."):
    output_folder = Path(output_folder)

    metrics = []
    for result in history:
        metrics.append(result.result)

    metrics = pd.DataFrame(metrics)

    columns = list(metrics.columns)
    plots = {}

    if len(columns) > 1:
        for column in columns[1:]:
            sub_metrics = metrics[[column, columns[0]]]
            is_efficient = is_pareto(sub_metrics.to_numpy())

            efficient_points = sub_metrics[is_efficient]
            efficient_points = efficient_points.sort_values(column)

            plot = metrics.plot.scatter(x=column, y=columns[0])
            plot.plot(efficient_points[column], efficient_points[columns[0]], "-r")

            plots[column] = plot

    for title, plot in plots.items():
        plot.get_figure().savefig(output_folder / f"pareto_front_{title}.png")
        plot.get_figure().savefig(output_folder / f"pareto_front_{title}.pdf")
        tikzplotlib.save(output_folder / f"pareto_front_{title}.tikz")


def plot_history(history, output_folder="."):
    metrics = []
    for result in history:
        metrics.append(result.result)

    metrics = pd.DataFrame(metrics)
    # metrics.drop(columns=["acc_area", "acc_power"], inplace=True)

    print("plot multi")
    plot = plot_multi(metrics)
    if plot is None:
        return

    fig = plot.get_figure()
    fig.tight_layout()

    fig.savefig(output_folder / "history.png")
    fig.savefig(output_folder / "history.pdf")
    tikzplotlib.save(output_folder / "history.tikz")
