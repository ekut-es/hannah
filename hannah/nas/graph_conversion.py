import logging
import copy

from dataclasses import dataclass
from hannah.nas.performance_prediction.search_space.space import Conv1dEntity
from typing import Dict, Tuple, Any

import networkx as nx
import numpy as np

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
            weight_dtype_onehot = to_one_hot(quantizer.dtype, self.dtype_encondings)
            quant_bits = np.asarray([quantizer.bits], dtype=np.float32)
            method_onehot = (
                np.asarray([1.0, 0.0])
                if not quantizer.power_of_2
                else np.asarray([0.0, 1.0])
            )
            quant_attrs = np.hstack([weight_dtype_onehot, quant_bits, method_onehot])
        else:
            quant_attrs = np.hstack(
                [
                    to_one_hot("float", self.dtype_encondings),
                    np.asarray([32.0]),
                    np.asarray([0.0, 0.0]),
                ]
            )

        return quant_attrs

    def add_nodes_quantize(self, target, mod, args):
        type_onehot = to_one_hot("quantize", self.layer_encodings)
        quant_attrs = self.extract_quant_attrs(mod)
        features = np.hstack([type_onehot, quant_attrs])
        self.nx_graph.add_node(target, features=features)

        input_names = [
            arg.name if isinstance(arg, NamedTensor) else "unknown" for arg in args
        ]
        for input_name in input_names:
            self.nx_graph.add_edge(input_name, target)

        return target

    def add_nodes_relu(self, target, mod, args):
        type_onehot = to_one_hot("relu", self.layer_encodings)
        features = type_onehot

        self.nx_graph.add_node(target, features=features)

        input_names = [
            arg.name if isinstance(arg, NamedTensor) else "unknown" for arg in args
        ]
        for input_name in input_names:
            self.nx_graph.add_edge(input_name, target)

        return target

    def add_nodes_conv(self, target, mod, args):
        type_onehot = to_one_hot("conv", self.layer_encodings)
        bias = 1.0 if mod.bias else 0.0
        conv_attrs = np.asarray(
            [
                float(mod.in_channels),
                float(mod.out_channels),
                float(mod.kernel_size[0]),
                float(mod.stride[0]),
                float(mod.dilation[0]),
                float(mod.groups),
                bias,
            ],
            dtype=np.float32,
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

        name = target + "_conv"
        self.nx_graph.add_node(name, features=features)

        input_names = [
            arg.name if isinstance(arg, NamedTensor) else "unknown" for arg in args
        ]
        for input_name in input_names:
            self.nx_graph.add_edge(input_name, name)

        if type(mod) in [qat.ConvReLU1d, qat.ConvBnReLU1d]:
            relu_name = target + "_relu"
            relu_args = [NamedTensor(name, None)]
            self.add_nodes_relu(relu_name, None, relu_args)

            name = relu_name

        activation_post_process = getattr(mod, "activation_post_process", None)
        if activation_post_process:
            if not isinstance(activation_post_process, torch.nn.Identity):
                post_process_name = target + "_quant"
                post_process_args = [NamedTensor(name, None)]
                self.add_nodes_quantize(
                    post_process_name, activation_post_process, post_process_args
                )
                name = post_process_name

        return name  # Return Output Name

    def add_nodes_linear(self, target, mod, args):
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

        input_names = [
            arg.name if isinstance(arg, NamedTensor) else "unknown" for arg in args
        ]
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

        return name  # Return Output Name

    def add_nodes_pooling(self, target, mod, args):

        features = to_one_hot("global_avg_pool", self.layer_encodings)
        self.nx_graph.add_node(target, features=features)

        input_names = [
            arg.name if isinstance(arg, NamedTensor) else "unknown" for arg in args
        ]
        for input_name in input_names:
            self.nx_graph.add_edge(input_name, target)

        return target

    def add_nodes_add(self, target, mod, args):
        features = to_one_hot("add", self.layer_encodings)
        self.nx_graph.add_node(target, features=features)

        input_names = [
            arg.name if isinstance(arg, NamedTensor) else "unknown" for arg in args
        ]
        for input_name in input_names:
            self.nx_graph.add_edge(input_name, target)

        return target

    def call_function(self, target, args: Tuple, kwargs: Dict) -> Any:

        target_name = target.__name__ + str(self.func_num)
        if target.__name__ in self.conversions:
            output = self.conversions[target.__name__](target_name, None, args)

        self.func_num += 1
        args = [arg.tensor if isinstance(arg, NamedTensor) else arg for arg in args]
        return NamedTensor(output, super().call_function(target, args, kwargs))

    def call_method(self, target, args: Tuple, kwargs: Dict) -> Any:

        args = [arg.tensor if isinstance(arg, NamedTensor) else arg for arg in args]
        return super().call_method(target, args, kwargs)

    def call_module(self, target, args: Tuple, kwargs: Dict) -> Any:
        # print(target, args, kwargs)

        submod = self.fetch_attr(target)
        if type(submod) in self.conversions:

            output = self.conversions[type(submod)](target, submod, args)

        else:
            assert len(args) == 1
            output = args[0].name
            logging.error("No conversion for type %s", type(submod))

        args = [arg.tensor if isinstance(arg, NamedTensor) else arg for arg in args]
        return NamedTensor(output, super().call_module(target, args, kwargs))

    def placeholder(self, target, args, kwargs):
        layer_onehot = to_one_hot("placeholder", self.layer_encodings)
        tensor = super().placeholder(target, args, kwargs)
        dimension = np.asarray(tensor.shape, dtype=np.float32)

        quantizer = getattr(self.module, "activation_post_process", None)
        quant_attr = self.extract_quant_attrs(quantizer)
        features = np.hstack([layer_onehot, quant_attr, dimension])
        self.nx_graph.add_node(target, features=features)
        return NamedTensor(target, tensor)


def model_to_graph(model):
    tracer = GraphConversionTracer()

    model = copy.deepcopy(model)

    model.cpu()
    model.eval()
    traced_graph = tracer.trace(model.model)
    # print(traced_graph)
    interpreter = GraphConversionInterpreter(
        torch.fx.GraphModule(model.model, traced_graph)
    )
    result = interpreter.run(model.example_feature_array)

    # print(interpreter.nx_graph)

    # for edge in interpreter.nx_graph.edges:
    #     print(edge)

    return interpreter.nx_graph
