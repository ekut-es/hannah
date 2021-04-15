import logging

import torch.fx
from tvm.relay import op

from . import qat

try:
    import tvm.relay as relay
    import tvm
except ModuleNotFoundError:
    relay = None
    tvm = None


class QuantizationTracer(torch.fx.Tracer):

    LEAF_MODULES = [
        qat.Conv1d,
        qat.Conv2d,
        qat.ConvBn1d,
        qat.ConvBn2d,
        qat.ConvBnReLU1d,
        qat.ConvBnReLU2d,
        qat.ConvReLU1d,
        qat.ConvReLU2d,
    ]

    def is_leaf_module(self, module, module_qualified_name):
        for leaf_cls in self.LEAF_MODULES:
            if isinstance(module, leaf_cls):
                return True

        return super().is_leaf_module(module, module_qualified_name)


class RelayConverter(torch.fx.Interpreter):
    def __init__(self, graph_module):
        super().__init__(graph_module)

        if relay is None:
            raise Exception(
                "TVM does not seem to be installed, please make sure that 'import tvm.relay works'"
            )

        self.tvm_mod = None
        self.modules = {}
        for name, module in graph_module.named_modules():
            self.modules[name] = module

        self.outputs = {}
        self.func_args = []
        self.returns = []

        self.module_map = {
            qat.Conv1d: self._handle_qat_conv,
            qat.Conv2d: self._handle_qat_conv,
            qat.ConvBn1d: self._handle_qat_conv,
            qat.ConvBn2d: self._handle_qat_conv,
            qat.ConvBnReLU1d: self._handle_qat_conv,
            qat.ConvBnReLU2d: self._handle_qat_conv,
        }

    def _handle_qat_conv(self, module, result):
        weight = module.weight
        bias = module.bias

        if hasattr(module, "bn"):
            weight, bias = torch.nn.utils.fuse_batch_norm(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.padding_mode,
            )

        padding = tuple(module.padding)
        strides = tuple(module.strides)
        dilation = tuple(module.dilation)
        out_channels = module.channels
        in_channels = module.size[1]

        weight_quant = module.weight_fake_quant
        activation_quant = module.activation_post_process
        bias_quant = module.activation_post_process

        if module.bias_fake_quant:
            bias_quant = module.bias_fake_quant

        quant_weight = module.weight_fake_quant.quantize(weight)
        quant_bias = module.bias_fake_quant.quantize(bias)

        print(quantize)

        return True

    def _handle_module(self, node, result):
        module = self.modules[node.target]
        if type(module) in self.module_map:
            self.module_map[type(module)](module, result)
        else:
            raise Exception(f"Support for module: {module} is not supported")

    def _handle_placeholder(self, node, result):
        var = relay.var(node.name, relay.TensorType(result.shape))
        self.outputs[node.name] = var
        self.func_args.append(var)

    def _handle_output(self, node, result):
        print(node.target)

    def run_node(self, node):
        result = super().run_node(node)

        print(node, node.op, node.args, node.kwargs, node.type)

        if node.op == "call_module":
            self._handle_module(node, result)
        elif node.op == "output":
            self._handle_output(node, result)
        elif node.op == "placeholder":
            self._handle_placeholder(node, result)
        else:
            raise Exception(f"Node {node} with op {node.op} is not supported")

        return result

    def propagate(self, *args):
        return super().run(*args)

    def run(self, input):
        tvm_mod = tvm.IRModule()

        super().run(input)

        ret = relay.const(1, dtype="int8")
        function = relay.Function(self.func_args, ret)
        tvm_mod["main"] = function

        print(tvm_mod)

        return tvm_mod
