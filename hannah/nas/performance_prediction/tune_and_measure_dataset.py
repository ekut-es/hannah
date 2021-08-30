import numpy as np

from search_space.space import NetworkSpace, NetworkEntity, point2knob
from search_space.tune import tune
from search_space.execute import execute_and_measure
from tvm import auto_scheduler
import os
from features import graph_conversion
import csv
import pandas as pd
from pathlib import Path
from hannah_tvm import measure
import hydra
import multiprocessing

from hannah_tvm.config import Board


def tune_and_measure(config):
    # Prepare paths for data storage
    data_name = config.net.name + "_tuned_" + config.board.name
    board = Board(**config.board)
    Path(
        config.setup.working_directory + "data/{}/graph_defs/".format(data_name)
    ).mkdir(parents=True, exist_ok=True)
    Path(config.setup.working_directory + "data/{}/logs/".format(data_name)).mkdir(
        parents=True, exist_ok=True
    )

    cfg_space = NetworkSpace()
    cfg_space.from_yaml(
        config.setup.working_directory + "configs/" + config.net.name + ".yaml"
    )
    prop_file = (
        config.setup.working_directory
        + "data/{}/graph_defs/graph_properties.csv".format(data_name)
    )

    # Check which nets (idxs) in the search space are already processed
    if os.path.exists(prop_file):
        properties = pd.read_csv(prop_file)
        ids = list(properties["graph_id"])
    else:
        ids = []
    available = np.arange(np.prod(cfg_space.dims()) - 1, dtype=int)
    available = [x for x in available if x not in ids]
    idxes = np.random.choice(available, size=len(available))

    if config.board.local:
        measure_ctx = auto_scheduler.LocalRPCMeasureContext()
    else:
        measure_ctx = measure.AutomateRPCMeasureContext(board)

    for idx in idxes:
        print("{}|{}".format(idx, np.prod(cfg_space.collapsed_dims())))
        cfg = point2knob(idx, cfg_space.collapsed_dims())
        print("CFG:", cfg)
        log_file = config.setup.working_directory + "data/{}/logs/log.json".format(
            data_name
        )
        conf_log = config.setup.working_directory + "data/{}/logs/{}.yaml".format(
            data_name, idx
        )
        task_log = config.setup.working_directory + "data/{}/logs/tasks.yaml".format(
            data_name
        )
        input_shape = tuple(config.net.input_shape)
        print(type(input_shape), input_shape)

        # Instantiate net and convert to graph
        net = NetworkEntity(cfg_space, cfg_space.expand_config(cfg))
        graph = graph_conversion.to_dgl_graph(net)
        edges_src = [edge.item() for edge in graph.edges()[0]]
        edges_dst = [edge.item() for edge in graph.edges()[1]]
        rows = [[idx, s, d] for s, d in zip(edges_src, edges_dst)]
        header = ["graph_id", "src", "dst"]
        edge_file = (
            config.setup.working_directory
            + "data/{}/graph_defs/graph_edges.csv".format(data_name)
        )
        file_exists = os.path.exists(edge_file)

        with open(edge_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerows(rows)
        print("Saved graph data ...")

        # tune & measure
        try:
            tune(
                net,
                input_shape,
                board,
                measure_ctx,
                log_file=log_file,
                config_log=conf_log,
                task_log=task_log,
            )
            prof_res = execute_and_measure(
                net, input_shape, board, measure_ctx, log_file
            )
            cost = np.mean(prof_res)
        except Exception as e:
            print("Exception in tuning or execution")
            print(str(e))
            cost = 100000

        # save results
        row = [idx, cost, len(graph.nodes())]
        header = ["graph_id", "label", "num_nodes"]
        file_exists = os.path.exists(prop_file)
        with open(prop_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

    for p in multiprocessing.active_children():
        print("Terminate process", p.name)
        p.terminate()


@hydra.main(config_name="conf", config_path="conf")
def main(config):
    print(config)
    return tune_and_measure(config)


if __name__ == "__main__":
    main()
    # tune_and_measure(None)
