import sys
from typing import Any, Text, Iterable, List, Dict, Sequence, Optional, Tuple, Union
from collections import defaultdict, namedtuple
from functools import reduce


from torch.utils.data import DataLoader

import onnx
from onnx import optimizer
from onnx import utils
from onnx import numpy_helper, ValueInfoProto, AttributeProto, GraphProto, NodeProto, TensorProto, TensorShapeProto

import numpy as np
from copy import deepcopy, copy
from itertools import product

from .config import ConfigBuilder
from . import dataset

def _convertAttributeProto(onnx_arg): 
    """
    Convert an ONNX AttributeProto into an appropriate Python object
    for the type.
    NB: Tensor attribute gets returned as numpy array
    """
    if onnx_arg.HasField('f'):
        return onnx_arg.f
    elif onnx_arg.HasField('i'):
        return onnx_arg.i
    elif onnx_arg.HasField('s'):
        return onnx_arg.s
    elif onnx_arg.HasField('t'):
        return numpy_helper.to_array(onnx_arg.t)
    elif len(onnx_arg.floats):
        return list(onnx_arg.floats)
    elif len(onnx_arg.ints):
        return list(onnx_arg.ints)
    elif len(onnx_arg.strings):
        return list(onnx_arg.strings)
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(onnx_arg))


EdgeInfo = namedtuple('EdgeInfo', ['name', 'type', 'shape'])
    
def _input_from_onnx_input(input) -> EdgeInfo: 
    name = input.name
    type = input.type.tensor_type.elem_type
    shape = tuple([d.dim_value for d in input.type.tensor_type.shape.dim])
    return EdgeInfo(name, type, shape)

class Attributes(Dict[Text, Any]):
    @staticmethod
    def from_onnx(args : AttributeProto) -> Any:  
        d = Attributes()
        for arg in args:
            d[arg.name] = _convertAttributeProto(arg)
        return d

class ComputeNode(object):

    def __init__(self, name : str,
                 op_type : str,
                 attrs : Attributes,
                 inputs : List[str],
                 outputs : List[str]) -> None: 
        self.name : str = name
        self.op_type : str = op_type
        self.attrs : Attributes = attrs
        self.inputs : List[str] = inputs
        self.outputs : List[str] = outputs

        self.input_tensors : Dict[str, np.ndarray] = {}  
        self.parents : List[ComputeNode] = []  
        self.children : List[ComputeNode]  = []  
        self.metadata : Dict[Any, Any] = {} 
    
    @staticmethod
    def from_onnx(node) -> Any:  
        attrs = Attributes.from_onnx(node.attribute)
        name = Text(node.name)
        if len(name) == 0:
            name = node.op_type + "_".join(node.output)
        return ComputeNode(name, node.op_type, attrs, list(node.input), list(node.output))


class ComputeGraph(object):

    def __init__(self, nodes : List[ComputeNode], inputs, outputs, shape_dict):
        self.nodes : List[ComputeNode] = nodes
        self.inputs  = inputs
        self.outputs = outputs
        self.shape_dict : Dict[str, np.ndarray] = shape_dict

    @staticmethod
    def from_onnx(graph) -> Any:  
        input_tensors = {
            t.name: numpy_helper.to_array(t) for t in graph.initializer
        }
       
        nodes_ = []
        nodes_by_input : Dict[str, List[ComputeNode]]= {} 
        nodes_by_output: Dict[str, ComputeNode] = {}
        for node in graph.node:
            node_ = ComputeNode.from_onnx(node)
            for input_ in node_.inputs:
                if input_ in input_tensors:
                    node_.input_tensors[input_] = input_tensors[input_]
                else:
                    if input_ in nodes_by_input:
                        input_nodes = nodes_by_input[input_]
                    else:
                        input_nodes = []
                        nodes_by_input[input_] = input_nodes
                    input_nodes.append(node_)
            for output_ in node_.outputs:
                nodes_by_output[output_] = node_
            nodes_.append(node_)

        inputs = []
        for i in graph.input:
            if i.name not in input_tensors:
                inputs.append(_input_from_onnx_input(i))

        outputs = []
        for o in graph.output:
            outputs.append(_input_from_onnx_input(o))

        for node_ in nodes_:
            for input_ in node_.inputs:
                if input_ in nodes_by_output:
                    node_.parents.append(nodes_by_output[input_])
            for output_ in node_.outputs:
                if output_ in nodes_by_input:
                    node_.children.extend(nodes_by_input[output_])

        # Dictionary to hold the "value_info" field from ONNX graph
        shape_dict : Dict[Text, Any]= {}
        def extract_value_info(shape_dict, 
                               value_info, 
                               ):
           
            t = tuple([int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim])
            if t:
                shape_dict[value_info.name] = t

                
        for value_info in graph.value_info:
            extract_value_info(shape_dict, value_info)

        return ComputeGraph(nodes_, inputs, outputs, shape_dict)



def remove_node(graph, node):
    print("Removing node", node.name)
    if not node in graph.nodes:
        return

    graph.nodes.remove(node)

    parents = node.parents

    for parent in node.parents:
        parent.children.remove(node)
        if not parent.children:
            remove_node(graph, parent)

    for child in node.children:
        child.parents.remove(node)

def get_shape(name : Text, graph : ComputeGraph, node : ComputeNode) -> Iterable[int]:
    for input in graph.inputs:
        if input.name == name:
            return input.shape
        
    for output in graph.outputs:
        if output.name == name:
            return output.shape
    
    if name in node.input_tensors:
        return node.input_tensors[name].shape
    
    if name in graph.shape_dict:
        return graph.shape_dict[name]

    return ()

def is_input(name : Text, graph : ComputeGraph) -> bool:
    for input in graph.inputs:
        if input.name == name:
            return True

    return False

def is_output(name : Text, graph : ComputeGraph) -> bool:
    for output in graph.outputs:
        if output.name == name:
            return True

    return False


def reduce_mult(data : Iterable[int]) -> int:
    return reduce(lambda x, y: x * y, data, 1)

def export_model(config):
    onnx_model = onnx.load(config["input_file"])
    onnx.checker.check_model(onnx_model)


    ## Set input batch size to 0
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1

    ## Remove Dropouts
    for op_id, op in enumerate(onnx_model.graph.node):
        if op.op_type == "Dropout":
            op.attribute[0].f = 0.0

    #TODO: remove BatchNorm
        
    print("Running model optimization")
    optimized_model = optimizer.optimize(onnx_model, ["eliminate_nop_dropout"])
    optimized_model = utils.polish_model(optimized_model)

    onnx.save(optimized_model, "polished_model.onnx")

    graph = ComputeGraph.from_onnx(optimized_model.graph)

    print("Running constant propagation")
    constant_states = constant_propagation(graph)

    #Remove nodes with constant values
    for node in list(graph.nodes):
        is_constant = True
        print(node.name)
        for output in node.outputs:
           
            if constant_states[output].value is None:
                is_constant = False
                print(output, "is not constant")
                
        if is_constant:
            remove_node(graph, node)

    # Add shape information form constant propagation:
    for var, res in constant_states.items():
        if var in graph.shape_dict:
            shape = graph.shape_dict[var]
            if res.shape != shape:
                print("Warning: Shapes do not match: ", var, res.shape, shape)
                graph.shape_dict[var] = res.shape
        elif res.shape is not None:
            graph.shape_dict[var] = res.shape
                
                
    # Generate Node Parameters
    parameter_header = "#ifndef NETWORK_PARAMETERS_H\n";
    parameter_header += "#define NETWORK_PARAMETERS_H\n";
    parameter_header += "#include \"pico-cnn/parameters.h\"\n\n"
    parameter_code = "#include \"network_parameters.h\"\n\n";
    for node in graph.nodes:
        if node.input_tensors:
            if node.op_type == "Conv" or node.op_type == "Gemm":
                type_code = "fp_t " + node.name + "_" + "coef[]"

                coef = node.input_tensors[node.inputs[1]]
                
                if node.op_type == "Gemm":
                    coef = coef.transpose()
                    
                declaration = "// " + str(coef.shape) + "\n"
                declaration += "extern " + type_code + ";"
                definition  = type_code + " = {" + ",".join((str(x) for x in coef.flatten())) + "};"

                parameter_code += definition + "\n"
                parameter_header += declaration + "\n"

            
                type_code = "fp_t " + node.name + "_" + "bias[]"

                bias = node.input_tensors[node.inputs[2]]
                declaration = "// " + str(bias.shape) + "\n"
                declaration += "extern " + type_code + ";"
                definition  = type_code + " = {" + ", ".join((str(x) for x in bias)) + "};"

                parameter_code += definition + "\n\n"
                parameter_header += declaration + "\n\n"
    parameter_header += "#endif \n"
                
    with open("network_parameters.h", "w") as f:
        f.write(parameter_header)

    with open("network_parameters.c", "w") as f:
        f.write(parameter_code)

        
    input_names = ["input"+str(name) for name, type, shape in graph.inputs]
    output_names = ["output"+str(name) for name, type, shape in graph.outputs]

    input_defs = ["fp_t *"+n for n in input_names];
    output_defs = ["fp_t *"+n for n in output_names];
    network_def = "void network(" + ", ".join(input_defs) + ", " + ", ".join(output_defs) +  ")"
    
    network_header = "#ifndef NETWORK_H\n"
    network_header += "#define NETWORK_H\n"
    network_header += "#include \"pico-cnn/parameters.h\"\n\n"
    network_header += network_def + ";\n"
    network_header += "#endif //NETWORK_H\n"

    network_code : Text =  "#include \"network.h\"\n"
    network_code += "#include \"network_parameters.h\"\n\n"
    network_code += "#include \"pico-cnn/pico-cnn.h\"\n\n"
    
    network_code += network_def+"{\n"

    implementation_code = ""
    buffer_code = ""
    buffer_code_end = ""
    
    for num, node in enumerate(graph.nodes):
        implementation_code += "  //Layer " + str(num) + " " +  node.name + " " +   node.op_type + "\n"
        implementation_code += "  //Attributes\n"
        for key, val in node.attrs.items():
            implementation_code += "  //  " + str(key) + ": " + str(val) + "\n"
        implementation_code += "  //Parameters\n"

        if node.op_type == "Conv":
            print("Generating convolution", node.name)

            attrs = node.attrs

            input_tensor = ""
            
            # Get Input Size
            input_id = node.inputs[0]
            input_shape = get_shape(input_id, graph, node)
            input_buffer = "buffer" + input_id
            if is_input(input_id, graph):
                input_buffer = "input" + input_id
            input_size = input_shape[2]
            input_channels = input_shape[1]
            
            output_buffer = "buffer" + node.outputs[0]
            if is_output(node.outputs[0], graph):
                output_buffer = "output" + node.outputs[0]

            output_shape = get_shape(node.outputs[0], graph, node)
            output_channels = output_shape[1]
            output_size = output_shape[2]
                
            coef_buffer = node.name + "_coef"
            coef_size = attrs["kernel_shape"][0]
            assert len(attrs["kernel_shape"]) == 1
            stride_size = attrs["strides"][0]
            assert attrs["pads"][0] == attrs["pads"][1]
            padding_size = attrs["pads"][0]
            
            bias_buffer = node.name + "_bias"

            dilation_size = attrs["dilations"][0]

            input_buffer_size = reduce_mult(input_shape)
            if not is_input(input_id, graph):
                buffer_code += "  static fp_t {buffer_name}[{buffer_size}];\n".format(buffer_size=input_buffer_size, buffer_name=input_buffer)

            
            implementation_code += """
  for(int i = 0; i < {output_channels}; i++){{
    convolution1d_naive(&({input_buffer}[0]), {input_size}, 
                        &({output_buffer}[i*{output_size}]),
                        &{coef_buffer}[i*{input_channels}*{coef_size}], {coef_size},
                        {stride_size}, 
                        {padding_size},
                        {bias_buffer}[i] / {input_channels}.0f,
                        {dilation_size});
    for(int j = 1; j < {input_channels}; j++){{
      static fp_t temp_buffer[{output_size}];

      convolution1d_naive(&({input_buffer}[j*{input_size}]), 
                          {input_size}, 
                          temp_buffer,
                          &{coef_buffer}[i*{input_channels}*{coef_size}+j*{coef_size}], 
                          {coef_size},
                          {stride_size}, 
                          {padding_size},
                          {bias_buffer}[i] / {input_channels}.0f,
                          {dilation_size});

      add_image2d_naive(&({output_buffer}[i*{output_size}]), temp_buffer, 1, {input_size});
    }}
  }}
""".format(input_buffer=input_buffer, input_size=input_size,
           input_channels=input_channels,
           output_buffer=output_buffer, output_size=output_size,
           output_channels=output_channels,
           coef_buffer=coef_buffer, coef_size=coef_size,
           stride_size=stride_size,
           padding_size=padding_size,
           bias_buffer=bias_buffer,
           dilation_size=dilation_size)


            
            
        elif node.op_type == "Gemm":
            print("generating fully connected layer")

            attrs = node.attrs

            assert node.attrs['alpha'] == 1.0
            assert node.attrs['beta'] == 1.0
            assert node.attrs['transB'] == 1

            input_id = node.inputs[0]
            input_shape = get_shape(input_id, graph, node)
            input_buffer = "buffer" + input_id
            if is_input(input_id, graph):
                input_buffer = "input" + input_id
            input_size = input_shape[1]

            output_buffer = "buffer" + node.outputs[0]
            if is_output(node.outputs[0], graph):
                output_buffer = "output" + node.outputs[0]
            output_shape = get_shape(node.outputs[0], graph, node)
            output_size = output_shape[1]

            input_buffer_size = reduce_mult(input_shape)
            if not is_input(input_id, graph):
                buffer_code += "  static fp_t {buffer_name}[{buffer_size}];\n".format(buffer_size=input_buffer_size, buffer_name=input_buffer)
            
            
            coef_buffer = node.name + "_coef"
            bias_buffer = node.name + "_bias"
            
            implementation_code += """
  fully_connected_naive({input_buffer}, {input_size}, 
                        {output_buffer}, {output_size},
                        {coef_buffer}, {bias_buffer});
""".format(input_buffer=input_buffer, input_size=input_size,
           output_buffer=output_buffer, output_size=output_size,
           coef_buffer=coef_buffer, bias_buffer=bias_buffer)
            
        elif node.op_type == "MaxPool":
            print("generating max pooling layer")

            attrs=node.attrs

            input_id = node.inputs[0]
            input_shape = get_shape(input_id, graph, node)
            input_buffer = "buffer" + input_id
            if is_input(input_id, graph):
                input_buffer = "input" + input_id
            input_width = input_shape[2]
            input_channels = input_shape[1]

            output_buffer = "buffer" + node.outputs[0]
            if is_output(node.outputs[0], graph):
                output_buffer = "output" + node.outputs[0]

            output_width = get_shape(node.outputs[0], graph, node)[2]
                
            assert len(attrs["kernel_shape"]) == 1
            kernel_size = attrs["kernel_shape"][0]
            kernel_stride = attrs["strides"][0]
            assert tuple(attrs["pads"]) == (0, 0)

            input_buffer_size = reduce_mult(input_shape)
            if not is_input(input_id, graph):
                buffer_code += "  static fp_t {buffer_name}[{buffer_size}];\n".format(buffer_size=input_buffer_size, buffer_name=input_buffer)
            
            
            implementation_code += """
  for(int i = 0; i < {input_channels}; i++){{
    max_pooling1d_naive(&{input_buffer}[i*{input_width}], {input_width}, 
                        &{output_buffer}[i*{output_width}],
                        {kernel_size},
                        {kernel_stride});
  }}
""".format(input_buffer=input_buffer, input_width=input_width, input_channels=input_channels,
           output_buffer=output_buffer, output_width=output_width,
           kernel_size=kernel_size,
           kernel_stride=kernel_stride)
            
            
        elif node.op_type == "Relu":
            print("generating relu layer")

            input_id = node.inputs[0]
            input_shape = get_shape(input_id, graph, node)
            input_buffer = "buffer" + input_id
            if is_input(input_id, graph):
                input_buffer = "input" + input_id
            input_width =  reduce_mult(input_shape) 
            input_height = 1
            
            output_buffer = "buffer" + node.outputs[0]
            if is_output(node.outputs[0], graph):
                output_buffer = "output" + node.outputs[0]


            input_buffer_size = reduce_mult(input_shape)
            if not is_input(input_id, graph):
                buffer_code += "  static fp_t {buffer_name}[{buffer_size}];\n".format(buffer_size=input_buffer_size, buffer_name=input_buffer)
            
            
            implementation_code += """
  relu_naive({input_buffer}, {input_height}, {input_width}, {output_buffer});
""".format(input_buffer = input_buffer, output_buffer=output_buffer,
           input_width=input_width, input_height=input_height)
            
        else:
            print("Unhandled node type:", node.op_type)
            print("Assuming NOP")


            input_id = node.inputs[0]
            input_shape = get_shape(input_id, graph, node)
            input_buffer = "buffer" + input_id
            if is_input(input_id, graph):
                input_buffer = "input" + input_id
            
            output_buffer = "buffer" + node.outputs[0]
            if is_output(node.outputs[0], graph):
                output_buffer = "output" + node.outputs[0]

            buffer_code_end += "  static fp_t *{input_name} = {output_name};\n".format(input_name=input_buffer, output_name=output_buffer)
        
        implementation_code += "\n"

    buffer_code += buffer_code_end
        
    network_code += buffer_code
    network_code += implementation_code
        
    network_code += "}\n"

    with open("network.c", "w") as f:
        f.write(network_code)

    with open("network.h", "w") as f:
        f.write(network_header)

def export_data(config):
    print("Exporting_input_data")
    train_set, dev_set, test_set = dataset.SpeechDataset.splits(config)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    data, label = next(iter(test_loader))
    print(data.shape, label)
    data = data.numpy().flatten()
    
    data_code = "#ifndef INPUT_DATA_H\n"
    data_code += "#include \"pico-cnn/parameters.h\"\n\n"
    data_code += "fp_t input[] = {" + ",".join((str(x) for x in data)) + "};\n"
    data_code += "#endif //INPUT_DATA_H\n"
    with open("input_data.h", "w") as f:
        f.write(data_code)
    
def main():
    global_config = dict(seed=0, input_file="", output_dir=".", cache_size=31288)
    builder = ConfigBuilder(
        dataset.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    config = builder.config_from_argparse(parser)

    export_model(config)
    export_data(config)

    return 0
        
if __name__ == "__main__":
    ret = main()
    sys.exit(ret)
