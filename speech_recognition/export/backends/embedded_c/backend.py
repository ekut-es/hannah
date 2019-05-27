from .compute_graph import *
from .constprop import constant_propagation
from .utils import reduce_mult

import onnx.backend.base as backend_base
import onnx

class BackendRep(backend_base.BackendRep):
    def __init__(self, onnx_model):
        self.onnx_model = onnx_model 
        self.network_code = ""
        self.network_header = ""

        self._export_model()

    def _remove_constants(self, graph, constant_states):
        #Remove nodes with constant values
        for node in list(graph.nodes):
            is_constant = True
            print(node.name)
            for output in node.outputs:
               
                if constant_states[output].value is None:
                    is_constant = False
                    
            if is_constant:
                graph.remove_node(node)
               
        
    def _export_model(self):
        graph = ComputeGraph.from_onnx(self.onnx_model.graph)
     
        print("Running constant propagation")
        constant_states = constant_propagation(graph)

        self._remove_constants(graph, constant_states)
        
         
        # Add shape information form constant propagation:
        for var, res in constant_states.items():
            if var in graph.shape_dict:
                shape = graph.shape_dict[var]
                if res.shape != shape:
                    print("Warning: Shapes do not match: ", var, res.shape, shape)
                    if res.shape is not None:
                        graph.shape_dict[var] = res.shape
            elif res.shape is not None:
                graph.shape_dict[var] = res.shape
                    
     
        # Remove nop nodes
        removed_nops = []
        for node in graph.nodes:
            if node.op_type == "Reshape":
                reshape_state = constant_states[node.inputs[1]]
                if (reshape_state.value == [1, -1]).all():
                    removed_input = node.inputs[0]
                    output = node.outputs[0]
                    removed_nops.append(node)
                    
                    for node in graph.nodes:
                        for num, input in enumerate(node.inputs):
                            if input == output:
                                print("node", node.name, "replacing input", input, "with", removed_input)
                                node.inputs[num] = removed_input
     
        for node in removed_nops:
            print("Removing node:", node.name)
            graph.nodes.remove(node)
     
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
     

                    if len(node.inputs) >= 3:
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
     
        memory_manager = None
        #memory_manager = MemoryManager()
        allocated_buffers = {}
        
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
                input_shape = graph.get_shape(input_id, node)
                input_buffer = "buffer" + input_id
                if graph.is_input(input_id):
                    input_buffer = "input" + input_id
                input_size = input_shape[2]
                input_channels = input_shape[1]
                
                output_buffer = "buffer" + node.outputs[0]
                if graph.is_output(node.outputs[0]):
                    output_buffer = "output" + node.outputs[0]
     
                output_shape = graph.get_shape(node.outputs[0], node)
                output_channels = output_shape[1]
                output_size = output_shape[2]
                    
                coef_buffer = node.name + "_coef"
                coef_size = attrs["kernel_shape"][0]
                assert len(attrs["kernel_shape"]) == 1 or attrs["kernel_shape"][1] == 1
                stride_size = attrs["strides"][0]
                print(attrs["pads"])
                #assert attrs["pads"][0] == attrs["pads"][1]
                padding_size = attrs["pads"][0]
                
                bias_buffer = node.name + "_bias"
     
                dilation_size = attrs["dilations"][0]
     
                #Allocate output
                if not graph.is_output(node.outputs[0]):
                    if memory_manager:
                        output_buffer = memory_manager.allocate_memory(reduce_mult(output_shape))
                        allocated_buffers[node.outputs[0]] = output_buffer
     
                        output_buffer = "(&global_buffer[{}])".format(output_buffer["start"])
                        
                input_buffer_size = reduce_mult(input_shape)
                if not graph.is_input(input_id):
                    if memory_manager:
                        input_buffer = allocated_buffers[node.inputs[0]]
                        memory_manager.free_memory(input_buffer)
                        input_buffer = "(&global_buffer[{}])".format(input_buffer["start"])
                    else:
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
                input_shape = graph.get_shape(input_id, node)
                input_buffer = "buffer" + input_id
                if graph.is_input(input_id):
                    input_buffer = "input" + input_id
                input_size = input_shape[1]
     
                output_buffer = "buffer" + node.outputs[0]
                if graph.is_output(node.outputs[0]):
                    output_buffer = "output" + node.outputs[0]
                output_shape = graph.get_shape(node.outputs[0], node)
                output_size = output_shape[1]
     
                input_buffer_size = reduce_mult(input_shape)
                if not graph.is_input(input_id):
                    if not memory_manager:
                        buffer_code += "  static fp_t {buffer_name}[{buffer_size}];\n".format(buffer_size=input_buffer_size, buffer_name=input_buffer)
     
                coef_buffer = node.name + "_coef"
                bias_buffer = node.name + "_bias"
     
                if memory_manager:
                    if not graph.is_output(node.outputs[0]):
                        output_buffer = memory_manager.allocate_memory(reduce_mult(output_shape))
                        allocated_buffers[node.outputs[0]] = output_buffer
                        output_buffer = "(&global_buffer[{}])".format(output_buffer["start"])
         
                    if not graph.is_input(node.inputs[0]):
                        input_buffer = allocated_buffers[node.inputs[0]]
                        memory_manager.free_memory(input_buffer)
                        del allocated_buffers[node.inputs[0]]
                        input_buffer = "(&global_buffer[{}])".format(input_buffer["start"])
                
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
                input_shape = graph.get_shape(input_id, node)
                input_buffer = "buffer" + input_id
                if graph.is_input(input_id):
                    input_buffer = "input" + input_id
                input_width = input_shape[2]
                input_channels = input_shape[1]
     
                output_buffer = "buffer" + node.outputs[0]
                if graph.is_output(node.outputs[0]):
                    output_buffer = "output" + node.outputs[0]
     
                output_width = graph.get_shape(node.outputs[0], node)[2]
                    
                assert len(attrs["kernel_shape"]) == 1
                kernel_size = attrs["kernel_shape"][0]
                kernel_stride = attrs["strides"][0]
                assert tuple(attrs["pads"]) == (0, 0)
     
                input_buffer_size = reduce_mult(input_shape)
                if not graph.is_input(input_id):
                    if not memory_manager:
                        buffer_code += "  static fp_t {buffer_name}[{buffer_size}];\n".format(buffer_size=input_buffer_size, buffer_name=input_buffer)
     
                if memory_manager:
                    output_shape = graph.get_shape(node.outputs[0], node)
                    if not graph.is_output(node.outputs[0]):
                        output_buffer = memory_manager.allocate_memory(reduce_mult(output_shape))
                        allocated_buffers[node.outputs[0]] = output_buffer
                        output_buffer = "(&global_buffer[{}])".format(output_buffer["start"])
                     
                    if not graph.is_input(node.inputs[0]):
                        input_buffer = allocated_buffers[node.inputs[0]]
                        memory_manager.free_memory(input_buffer)
                        del allocated_buffers[node.inputs[0]]
                        input_buffer = "(&global_buffer[{}])".format(input_buffer["start"])
                
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
                input_shape = graph.get_shape(input_id, node)
                input_buffer = "buffer" + input_id
                if graph.is_input(input_id):
                    input_buffer = "input" + input_id
                input_width =  reduce_mult(input_shape) 
                input_height = 1
                
                output_buffer = "buffer" + node.outputs[0]
                if graph.is_output(node.outputs[0]):
                    output_buffer = "output" + node.outputs[0]
     
     
                input_buffer_size = reduce_mult(input_shape)
                if not graph.is_input(input_id):
                    if not memory_manager:
                        buffer_code += "  static fp_t {buffer_name}[{buffer_size}];\n".format(buffer_size=input_buffer_size, buffer_name=input_buffer)
                
     
                if memory_manager:
                    output_shape = graph.get_shape(node.outputs[0], node)
                    if not graph.is_output(node.outputs[0]):
                        output_buffer = memory_manager.allocate_memory(reduce_mult(output_shape))
                        allocated_buffers[node.outputs[0]] = output_buffer
                        output_buffer = "(&global_buffer[{}])".format(output_buffer["start"])
                     
                    if not graph.is_input(node.inputs[0]):
                        input_buffer = allocated_buffers[node.inputs[0]]
                        memory_manager.free_memory(input_buffer)
                        del allocated_buffers[node.inputs[0]]
                        input_buffer = "(&global_buffer[{}])".format(input_buffer["start"])
                    
                implementation_code += """
      relu_naive({input_buffer}, {input_height}, {input_width}, {output_buffer});
    """.format(input_buffer = input_buffer, output_buffer=output_buffer,
               input_width=input_width, input_height=input_height)
                
            else:
                print("Unhandled node type:", node.op_type)
                print("Doing nothing")
     
                
            implementation_code += "\n"
     
        if memory_manager:
            buffer_code += "  fp_t global_buffer[{}];\n".format(memory_manager.max_memory)
            
        buffer_code += buffer_code_end
            
        network_code += buffer_code
        network_code += implementation_code
            
        network_code += "}\n"

        self.network_code = network_code
        self.network_header = network_header

    def save(self, folder):
        with open("network.c", "w") as f:
            f.write(self.network_code)
     
        with open("network.h", "w") as f:
            f.write(self.network_header)


class Backend(object):
    @classmethod
    def prepare(cls,
                model,  # type: ModelProto
                device='CPU',  # type: Text
                **kwargs  # type: Any
                ):  # type: (...) -> Optional[BackendRep]
        # TODO Remove Optional from return type
        onnx.checker.check_model(model)

        return BackendRep(model)
        
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
    
