import distiller
from distiller.data_loggers import PythonLogger, CsvLogger
import distiller.apputils as apputils
import logging
import pandas as pd
import torch
from functools import partial
from tabulate import tabulate
from collections import OrderedDict


def draw_classifier_to_file(model, png_fname, dummy_input, display_param_nodes=False, rankdir='TB', styles=None):
    """Draw a PyTorch classifier to a PNG file.  This a helper function that
    simplifies the interface of draw_model_to_file().

    Args:
        model: PyTorch model instance
        png_fname (string): PNG file name
        dummy_input (tensor): one batch of input_data
        display_param_nodes (boolean): if True, draw the parameter nodes
        rankdir: diagram direction.  'TB'/'BT' is Top-to-Bottom/Bottom-to-Top
                 'LR'/'R/L' is Left-to-Rt/Rt-to-Left
        styles: a dictionary of styles.  Key is module name.  Value is
                a legal pydot style dictionary.  For example:
                styles['conv1'] = {'shape': 'oval',
                                   'fillcolor': 'gray',
                                   'style': 'rounded, filled'}
    """

    msglogger = logging.getLogger()
    
    try:
        model = distiller.make_non_parallel_copy(model)
        dummy_input = dummy_input.to(distiller.model_device(model))
        g = distiller.SummaryGraph(model, dummy_input)
        distiller.draw_model_to_file(g, png_fname, display_param_nodes, rankdir, styles)
        print("Network PNG image generation completed")
    except FileNotFoundError:
        print("An error has occured while generating the network PNG image.")
        print("Please check that you have graphviz installed.")
        print("\t$ sudo apt-get install graphviz or")
        print("\t$ sudo yum install graphviz")


def model_summary(model, dummy_input, what):
    msglogger = logging.getLogger()
    
    if what == 'sparsity':
        pylogger = PythonLogger(msglogger)
        csvlogger = CsvLogger('weights.csv')

        distiller.log_weights_sparsity(model, -1, loggers=[pylogger, csvlogger])

    elif what == 'performance':
        df = distiller.model_performance_summary(model, dummy_input, dummy_input.shape[0])
        t = tabulate(df, headers='keys', tablefmt='psql', floatfmt=".5f")
        total_macs = df['MACs'].sum()
        total_acts = df['IFM volume'][0] + df['OFM volume'].sum()
        total_weights = df['Weights volume'].sum()
        estimated_acts = 2 * max(df['IFM volume'].max(), df['OFM volume'].max()) 
        msglogger.info("\n"+str(t))
        msglogger.info("Total MACs: " + "{:,}".format(total_macs))
        msglogger.info("Total Weights: " + "{:,}".format(total_weights))
        msglogger.info("Total Activations: " + "{:,}".format(total_acts))
        msglogger.info("Estimated Activations: " + "{:,}".format(estimated_acts))

        res = OrderedDict()
        res["Total MACs"] = total_macs
        res["Total Weights"] = total_weights
        res["Total Activations"] = total_acts
        res["Estimated Activations"] = estimated_acts

        return res
