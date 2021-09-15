import logging
import copy

from dataclasses import dataclass
from hannah.nas.performance_prediction.search_space.space import Conv1dEntity
from typing import Dict, Tuple, Any

import networkx as nx
import numpy as np

import math

import torch.fx
from hannah.models.factory import qat, qconfig, pooling


class GraphConversionTracer(torch.fx.Tracer):

    LEAF_MODULES = [
        qat.Conv1d,
        qat.Conv2d,
        qat.ConvBn1d,
        qat.ConvBn2d,
        qat.ConvBnReLU1d,
        qat.ConvBnReLU2d,
        qat.ConvReLU1d,
        qat.ConvReLU2d,
        qat.Linear,
        qat.LinearReLU,
        qconfig.STEQuantize,
        pooling.ApproximateGlobalAveragePooling1D,
        pooling.ApproximateGlobalAveragePooling2D,
    ]

    def is_leaf_module(self, module, module_qualified_name):
        for leaf_cls in self.LEAF_MODULES:
            if isinstance(module, leaf_cls):
                return True

        return super().is_leaf_module(module, module_qualified_name)


def to_one_hot(val, options):
    vec = np.zeros(len(options))
    options = np.array(options)
    try:
        index = np.where(val == options)[0][0]
        vec[index] = 1
    except Exception as e:
        pass
    return vec


@dataclass
class NamedTensor:
    name: str
    tensor: torch.Tensor
    quantization: Any = None


class GraphConversionInterpreter(torch.fx.Interpreter):
    def __init__(self, module, garbage_collect_values=True):
        super().__init__(module, garbage_collect_values)

        self.nx_graph = nx.DiGraph()

        self.conversions = {
            qat.ConvBnReLU1d: self.add_nodes_conv,
            qat.ConvBn1d: self.add_nodes_conv,
            qat.ConvReLU1d: self.add_nodes_conv,
            qat.Linear: self.add_nodes_linear,
            qat.LinearReLU: self.add_nodes_linear,
            qconfig.STEQuantize: self.add_nodes_quantize,
            pooling.ApproximateGlobalAveragePooling1D: self.add_nodes_pooling,
            torch.nn.ReLU: self.add_nodes_relu,
            "add": self.add_nodes_add,
        }
        self.layer_encodings = [
            "conv",
            "bn",
            "relu",
            "add",
            "quantize",
            "global_avg_pool",
            "linear",
            "placeholder",
        ]
        self.dtype_encondings = ["float", "int", "uint"]

        self.func_num = 0

    def extract_quant_attrs(self, quantizer):

        if quantizer:
            quant_attrs = {
                "dtype": quantizer.dtype,
                "bits": quantizer.bits,
                "method": "symmetric" if not quantizer.power_of_2 else "power_of_2",
            }
        else:
            quant_attrs = {"dtype": "float", "bits": 32, "method": "none"}

        return quant_attrs

    def extract_input_attrs(self, args):
        input_attrs = [
            {"name": arg.name, "shape": arg.tensor.shape, "quant": arg.quantization}
            for arg in args
        ]

        return input_attrs

    def add_nodes_quantize(self, target, mod, args, output):
        type_onehot = to_one_hot("quantize", self.layer_encodings)
        quant_attrs = self.extract_quant_attrs(mod)

        input_attrs = self.extract_input_attrs(args)
        self.nx_graph.add_node(
            target,
            attrs=quant_attrs,
            output={"quant": quant_attrs, "shape": output.shape},
            inputs=input_attrs,
            type="quantize",
        )

        input_names = [arg.name for arg in args]
        for input_name in input_names:
            self.nx_graph.add_edge(input_name, target)

        return NamedTensor(target, output, quantization=quant_attrs)

    def add_nodes_relu(self, target, mod, args, output):
        type_onehot = to_one_hot("relu", self.layer_encodings)
        features = type_onehot

        self.nx_graph.add_node(target, features=features)

        input_names = [arg.name for arg in args]
        for input_name in input_names:
            self.nx_graph.add_edge(input_name, target)

        return NamedTensor(target, output)

    def add_nodes_conv(self, target, mod, args, output):
        attrs = {}

        attrs["in_channels"] = mod.in_channels
        attrs["out_channels"] = mod.out_channels
        attrs["kernel_size"] = mod.kernel_size
        attrs["stride"] = mod.stride
        attrs["dilation"] = mod.dilation
        attrs["groups"] = mod.groups
        attrs["padding"] = mod.padding

        weight_quant_attrs = self.extract_quant_attrs(
            getattr(mod, "weight_fake_quant", None)
        )
        weight_attrs = {"quant": weight_quant_attrs, "shape": mod.weight.shape}

        bias_attrs = None
        if mod.bias is not None or hasattr(mod, "bn"):
            bias_quant_attrs = self.extract_quant_attrs(
                getattr(mod, "bias_fake_quant", None)
            )
            if mod.bias is not None:
                bias_shape = mod.bias.shape
            else:
                bias_shape = mod.bn.bias.shape

            bias_attrs = {"quant": bias_quant_attrs, "shape": bias_shape}

        name = target + "_conv"
        input_attrs = self.extract_input_attrs(args)
        output_quant = {"dtype": "float", "bits": 32, "method": "none"}
        if (
            input_attrs[0]["quant"]["dtype"] != "float"
            and weight_attrs["quant"]["dtype"] != "float"
        ):
            output_bits = (
                input_attrs[0]["quant"]["bits"]
                + weight_attrs["quant"]["bits"]
                + math.ceil(
                    math.log(
                        attrs["in_channels"] / attrs["groups"]
                        + sum(attrs["kernel_size"])
                    )
                )
            )
            output_quant = {
                "dtype": input_attrs[0]["quant"]["dtype"],
                "bits": output_bits,
                "method": input_attrs[0]["quant"]["method"],
            }
            output_attr = {"name": name, "quant": output_quant, "shape": output.shape}
        self.nx_graph.add_node(
            name,
            attrs=attrs,
            type="conv",
            weight=weight_attrs,
            bias=bias_attrs,
            inputs=input_attrs,
            output=output_attr,
        )

        input_names = [arg.name for arg in args]
        for input_name in input_names:
            self.nx_graph.add_edge(input_name, name)

        if type(mod) in [qat.ConvReLU1d, qat.ConvBnReLU1d]:
            relu_name = target + "_relu"
            relu_args = [NamedTensor(name, output, quantization=output_quant)]
            self.add_nodes_relu(relu_name, None, relu_args, output)

            name = relu_name

        quantization = None
        activation_post_process = getattr(mod, "activation_post_process", None)
        if activation_post_process:
            if not isinstance(activation_post_process, torch.nn.Identity):
                post_process_name = target + "_quant"
                post_process_args = [
                    NamedTensor(name, output, quantization=output_quant)
                ]
                quant_out = self.add_nodes_quantize(
                    post_process_name,
                    activation_post_process,
                    post_process_args,
                    output,
                )
                quantization = quant_out.quantization
                name = post_process_name

        return NamedTensor(name, output, quantization=quantization)

    def add_nodes_linear(self, target, mod, args, output):
        type_onehot = to_one_hot("linear", self.layer_encodings)
        bias = 1.0 if mod.bias else 0.0
        conv_attrs = np.asarray(
            [float(mod.in_features), float(mod.out_features), bias], dtype=np.float32
        )

        weight_quant_attrs = self.extract_quant_attrs(
            getattr(mod, "weight_fake_quant", None)
        )
        bias_quant_attrs = self.extract_quant_attrs(
            getattr(mod, "bias_fake_quant", None)
        )

        features = np.hstack(
            [type_onehot, conv_attrs, weight_quant_attrs, bias_quant_attrs]
        )

        name = target + "_linear"
        self.nx_graph.add_node(name, features=features)

        input_names = [arg.name for arg in args]
        for input_name in input_names:
            self.nx_graph.add_edge(input_name, name)

        if type(mod) in [qat.LinearReLU]:
            relu_name = target + "_relu"
            self.add_nodes_relu(relu_name, None)
            self.nx_graph.add_edge(name, relu_name)

            name = relu_name

        activation_post_process = getattr(mod, "activation_post_process", None)
        if activation_post_process:
            if not isinstance(activation_post_process, torch.nn.Identity):
                post_process_name = target + "_quant"
                self.add_nodes_quantize(post_process_name, activation_post_process)
                self.nx_graph.add_edge(name, post_process_name)
                name = post_process_name

        return NamedTensor(name, output)

    def add_nodes_pooling(self, target, mod, args, output):

        features = to_one_hot("global_avg_pool", self.layer_encodings)
        self.nx_graph.add_node(target, features=features)

        input_names = [arg.name for arg in args]
        for input_name in input_names:
            self.nx_graph.add_edge(input_name, target)

        return NamedTensor(target, output)

    def add_nodes_add(self, target, mod, args, output):
        features = to_one_hot("add", self.layer_encodings)
        self.nx_graph.add_node(target, features=features)

        input_names = [arg.name for arg in args]
        for input_name in input_names:
            self.nx_graph.add_edge(input_name, target)

        return NamedTensor(target, output)

    def call_function(self, target, args: Tuple, kwargs: Dict) -> Any:
        self.func_num += 1
        arg_tensors = [arg.tensor for arg in args]
        output_tensor = super().call_function(target, arg_tensors, kwargs)
        target_name = target.__name__ + str(self.func_num)
        if target.__name__ in self.conversions:
            output = self.conversions[target.__name__](
                target_name, None, args, output_tensor
            )

        return output

    def call_method(self, target, args: Tuple, kwargs: Dict) -> Any:

        args = [arg.tensor for arg in args]
        return super().call_method(target, args, kwargs)

    def call_module(self, target, args: Tuple, kwargs: Dict) -> Any:
        # print(target, args, kwargs)

        tensor_args = [arg.tensor for arg in args]
        output_tensor = super().call_module(target, tensor_args, kwargs)
        submod = self.fetch_attr(target)
        if type(submod) in self.conversions:

            output = self.conversions[type(submod)](target, submod, args, output_tensor)

        else:
            assert len(args) == 1
            output = args[0].name
            logging.error("No conversion for type %s", type(submod))

        return output

    def placeholder(self, target, args, kwargs):
        tensor = super().placeholder(target, args, kwargs)
        dimension = np.asarray(tensor.shape, dtype=np.float32)

        quantizer = getattr(self.module, "activation_post_process", None)
        quant_attr = self.extract_quant_attrs(quantizer)

        self.nx_graph.add_node(
            target, output={"quant": quant_attr, "shape": dimension}, type="placeholder"
        )
        return NamedTensor(target, tensor, quantization=quant_attr)


def model_to_graph(model, input):
    tracer = GraphConversionTracer()

    model = copy.deepcopy(model)

    model.cpu()
    model.eval()
    traced_graph = tracer.trace(model)
    # print(traced_graph)
    interpreter = GraphConversionInterpreter(torch.fx.GraphModule(model, traced_graph))
    result = interpreter.run(input)

    # print(interpreter.nx_graph)

    # for edge in interpreter.nx_graph.edges:
    #     print(edge)

    return interpreter.nx_graph
