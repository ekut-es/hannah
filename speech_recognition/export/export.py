import sys
from typing import Any, Text, Iterable, List, Dict, Sequence, Optional, Tuple, Union
from collections import defaultdict, namedtuple


from torch.utils.data import DataLoader

import onnx
from onnx import optimizer
from onnx import utils
from onnx import numpy_helper, ValueInfoProto, AttributeProto, GraphProto, NodeProto, TensorProto, TensorShapeProto

import numpy as np
from copy import deepcopy, copy
from itertools import product


from ..config import ConfigBuilder
from .. import dataset
        
from .backends.embedded_c import Backend

def main():
    global_config = dict(seed=0, input_file="", output_dir=".")
    builder = ConfigBuilder(
        dataset.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    config = builder.config_from_argparse(parser)

    
    onnx_model = onnx.load(config["input_file"])
    onnx.checker.check_model(onnx_model)

    ## Set input batch size to 1
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1

    ## Remove Dropouts
    for op_id, op in enumerate(onnx_model.graph.node):
        if op.op_type == "Dropout":
            op.attribute[0].f = 0.0
        
    print("Running model optimization")
    optimized_model = optimizer.optimize(onnx_model, ["eliminate_nop_dropout"])
    optimized_model = utils.polish_model(optimized_model)

    onnx.save(optimized_model, "polished_model.onnx")
    

    backend_model = Backend.prepare(optimized_model)
    #export_data(config)

    return 0
        
