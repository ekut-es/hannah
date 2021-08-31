import logging
import os 
import csv

from hannah.nas.graph_conversion import model_to_graph
from hannah.nas.performance_prediction.simple import to_dgl_graph

WD = './'
DATA_NAME = 'test'

# Alternatively as a generator?
def generate_network_models(exclude=[]):
    """Generates a list of models from a search space

    Parameters
    ----------
    exclude : list, optional
        model idxes to ignore, by default []

    Returns
    -------
    list
        list of models
    """        
    models: list = [None]
    return models


def tune_models(models):
    """Iterate through models, save graphs to file, tune & measure, 
        save measure results to file

    Parameters
    ----------
    models : list
        list of models to tune
    """    
    for idx, model in enumerate(models):
        nx_graph = model_to_graph(model)
        dgl_graph = to_dgl_graph(nx_graph)
        serialize_graph(dgl_graph, idx, WD, DATA_NAME)
        cost, std = run_tuning(model)
        serialize_measure_results(dgl_graph, idx, cost, std, WD, DATA_NAME)


def serialize_graph(dgl_graph, idx, wd, data_name):
    """Save DGL graph edges to file

    Parameters
    ----------
    dgl_graph : dgl.DGLGraph
        the graph to save
    idx : int
        graph_id or index of the graph that 
        uniquely describes it in the search space
    wd : str
        working directory
    data_name : str
        name of dataset
    """    
    edges_src = [edge.item() for edge in dgl_graph.edges()[0]]
    edges_dst = [edge.item() for edge in dgl_graph.edges()[1]]
    rows = [[idx, s, d] for s, d in zip(edges_src, edges_dst)]
    header = ["graph_id", "src", "dst"]
    edge_file = wd + "data/{}/graph_defs/graph_edges.csv".format(data_name)
    file_exists = os.path.exists(edge_file)
    with open(edge_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(rows)

        logging.info("Saved graph data")

def run_tuning(model):
    """Compile and tune model

    Parameters
    ----------
    model : [type]
        model to tune

    Returns
    -------
    (float, float)
        cost and standard dev
    """    
    return 0, 0


def serialize_measure_results(dgl_graph, idx, cost, std, wd, data_name):
    """Save measurement results to file

    Parameters
    ----------
    dgl_graph : dgl.DGLGraph
        Graph that the measurements belong to
    idx : int
        unique graph_id
    cost : float
        measured cost
    std : float
        measured standard deviation
    wd : str
        working dir
    data_name : str
        name of dataset
    """    
    row = [idx, cost, std, len(dgl_graph.nodes())]
    header = ["graph_id", "cost", 'std', "num_nodes"]
    prop_file = wd + "/data/{}/graph_defs/graph_properties.csv".format(data_name)
    file_exists = os.path.exists(prop_file)
    with open(prop_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

if __name__ == "__main__":
    logging.info("Start dataset generation")
    models = generate_network_models()
    tune_models(models)
    logging.info('Finished dataset generation')