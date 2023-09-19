#
# Copyright (c) 2023 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import logging
from collections import OrderedDict

import pandas as pd
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from tabulate import tabulate
from torch.fx.graph_module import GraphModule

from hannah.models.ofa.submodules.elasticBase import ElasticBase1d

from ..models.factory import qat
from ..models.ofa import OFAModel
from ..models.ofa.submodules.elastickernelconv import ConvBn1d, ConvBnReLu1d, ConvRelu1d
from ..models.ofa.type_utils import elastic_conv_type, elastic_Linear_type
from ..models.sinc import SincNet

import torch.fx as fx
from hannah.nas.graph_conversion import GraphConversionTracer
from hannah.nas.functional_operators.operators import conv2d, linear, add

msglogger = logging.getLogger(__name__)


def walk_model(model, dummy_input):
    """Adapted from IntelLabs Distiller

    Args:
      model:
      dummy_input:

    Returns:

    """

    data = {
        "Name": [],
        "Type": [],
        "Attrs": [],
        "IFM": [],
        "IFM volume": [],
        "OFM": [],
        "OFM volume": [],
        "Weights volume": [],
        "MACs": [],
    }

    def prod(seq):
        """

        Args:
          seq:

        Returns:

        """
        result = 1.0
        for number in seq:
            result *= number
        return int(result)

    def get_name_by_module(m):
        """

        Args:
          m:

        Returns:

        """
        for module_name, mod in model.named_modules():
            if m == mod:
                return module_name

    def collect(module, input, output):
        """

        Args:
          module:
          input:
          output:

        Returns:

        """
        # if len(list(module.children())) != 0:
        #    return
        try:
            if isinstance(output, tuple):
                output = output[0]

            volume_ifm = prod(input[0].size())
            volume_ofm = prod(output.size())
            extra = get_extra(module, volume_ofm, output)
            if extra is not None:
                weights, macs, attrs = extra
            else:
                weights, macs, attrs = 0, 0, 0
            data["Name"] += [get_name_by_module(module)]
            data["Type"] += [module.__class__.__name__]
            data["Attrs"] += [attrs]
            data["IFM"] += [tuple(input[0].size())]
            data["IFM volume"] += [volume_ifm]
            data["OFM"] += [tuple(output.size())]
            data["OFM volume"] += [volume_ofm]
            data["Weights volume"] += [int(weights)]
            data["MACs"] += [int(macs)]
        except Exception as e:
            msglogger.error("Could not get summary from %s", str(module))
            msglogger.error(str(e))

    def get_extra(module, volume_ofm, output):
        """

        Args:
          module:
          volume_ofm:
          output:

        Returns:

        """
        classes = {
            elastic_conv_type: get_elastic_conv,
            elastic_Linear_type: get_elastic_linear,
            ConvBn1d: get_conv,
            ConvRelu1d: get_conv,
            ConvBnReLu1d: get_conv,
            torch.nn.Conv1d: get_conv,
            torch.nn.Conv2d: get_conv,
            qat.Conv1d: get_conv,
            qat.Conv2d: get_conv,
            qat.ConvBn1d: get_conv,
            qat.ConvBn2d: get_conv,
            qat.ConvBnReLU1d: get_conv,
            qat.ConvBnReLU2d: get_conv,
            SincNet: get_sinc_conv,
            torch.nn.Linear: get_fc,
            qat.Linear: get_fc,
        }
        if type(module) in classes.keys():
            return classes[type(module)](module, volume_ofm, output)
        else:
            return get_generic(module)

    def get_conv_macs(module, volume_ofm):
        """

        Args:
          module:
          volume_ofm:

        Returns:

        """
        return volume_ofm * (
            module.in_channels / module.groups * prod(module.kernel_size)
        )

    def get_conv_attrs(module):
        """

        Args:
          module:

        Returns:

        """
        attrs = "k=" + "(" + (", ").join(["%d" % v for v in module.kernel_size]) + ")"
        attrs += ", s=" + "(" + (", ").join(["%d" % v for v in module.stride]) + ")"
        attrs += ", g=(%d)" % module.groups
        attrs += ", dsc=(%s)" % str(
            module.in_channels == module.out_channels == module.groups
        )
        attrs += ", d=" + "(" + ", ".join(["%d" % v for v in module.dilation]) + ")"
        return attrs

    def get_elastic_conv(module, volume_ofm, output):
        """

        Args:
          module:
          volume_ofm:
          output:

        Returns:

        """
        tmp = module.assemble_basic_module()
        return get_conv(tmp, volume_ofm, output)

    def get_elastic_linear(module, volume_ofm, output):
        """

        Args:
          module:
          volume_ofm:
          output:

        Returns:

        """
        tmp = module.assemble_basic_module()
        return get_fc(tmp, volume_ofm, output)

    def get_conv(module, volume_ofm, output):
        """

        Args:
          module:
          volume_ofm:
          output:

        Returns:

        """
        weights = (
            module.out_channels
            * module.in_channels
            / module.groups
            * prod(module.kernel_size)
        )
        macs = get_conv_macs(module, volume_ofm)
        attrs = get_conv_attrs(module)
        return weights, macs, attrs

    def get_sinc_conv(module, volume_ofm, output):
        """

        Args:
          module:
          volume_ofm:
          output:

        Returns:

        """
        weights = 2 * module.out_channels * module.in_channels / module.groups
        macs = get_conv_macs(module, volume_ofm)
        attrs = get_conv_attrs(module)
        return weights, macs, attrs

    def get_fc(module, volume_ofm, output):
        """

        Args:
          module:
          volume_ofm:
          output:

        Returns:

        """
        weights = macs = module.in_features * module.out_features
        attrs = ""
        return weights, macs, attrs

    def get_generic(module):
        """

        Args:
          module:

        Returns:

        """
        if isinstance(module, torch.nn.Dropout):
            return
        weights = macs = 0
        attrs = ""
        return weights, macs, attrs

    hooks = list()

    for name, module in model.named_modules():
        if module != model:
            hooks += [module.register_forward_hook(collect)]
    try:
        with torch.no_grad():
            _ = model(dummy_input)
    finally:
        for hook in hooks:
            hook.remove()

    df = pd.DataFrame(data=data)
    return df


class MacSummaryCallback(Callback):
    """ """

    def _do_summary(self, pl_module, input=None, print_log=True):
        """

        Args:
          pl_module:
          print_log:  (Default value = True)

        Returns:

        """
        dummy_input = input
        if dummy_input is None:
            dummy_input = pl_module.example_feature_array
        dummy_input = dummy_input.to(pl_module.device)

        total_macs = 0.0
        total_acts = 0.0
        total_weights = 0.0
        estimated_acts = 0.0
        model = pl_module.model
        ofamodel = isinstance(model, OFAModel)
        if ofamodel:
            if model.validation_model is None:
                model.build_validation_model()
            model = model.validation_model

        try:
            df = walk_model(model, dummy_input)
            if ofamodel:
                pl_module.model.reset_validation_model()
            t = tabulate(df, headers="keys", tablefmt="psql", floatfmt=".5f")
            total_macs = df["MACs"].sum()
            total_acts = df["IFM volume"][0] + df["OFM volume"].sum()
            total_weights = df["Weights volume"].sum()
            estimated_acts = 2 * max(df["IFM volume"].max(), df["OFM volume"].max())
            if print_log:
                msglogger.info("\n" + str(t))
                msglogger.info("Total MACs: " + "{:,}".format(total_macs))
                msglogger.info("Total Weights: " + "{:,}".format(total_weights))
                msglogger.info("Total Activations: " + "{:,}".format(total_acts))
                msglogger.info(
                    "Estimated Activations: " + "{:,}".format(estimated_acts)
                )
        except RuntimeError as e:
            if ofamodel:
                pl_module.model.reset_validation_model()
            msglogger.warning("Could not create performance summary: %s", str(e))
            return OrderedDict()

        res = OrderedDict()
        res["total_macs"] = total_macs
        res["total_weights"] = total_weights
        res["total_act"] = total_acts
        res["est_act"] = estimated_acts

        return res

    def predict(self, pl_module, input=input):
        """

        Args:
          pl_module:

        Returns:

        """

        res = self.estimate(pl_module, input=input)

        return res

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """

        Args:
          trainer:
          pl_module:

        Returns:

        """
        pl_module.eval()
        try:
            self._do_summary(pl_module)
        except Exception as e:
            msglogger.critical("_do_summary failed")
            msglogger.critical(str(e))
        pl_module.train()

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        """

        Args:
          trainer:
          pl_module:

        Returns:

        """
        pl_module.eval()
        self._do_summary(pl_module)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """

        Args:
          trainer:
          pl_module:

        Returns:

        """
        res = {}
        res = self._do_summary(pl_module, print_log=False)

        for k, v in res.items():
            pl_module.log(k, float(v), rank_zero_only=True)

    def estimate(self, pl_module, input=None):
        """Generate Summary Metrics for neural network

        Args:
          pl_module(pytorch_lightning.LightningModule): pytorch lightning module to summarize

        Returns:
          dict[str, float]: Dict of MetricName => Metric Value

        """
        pl_module.eval()
        res = {}
        try:
            res = self._do_summary(pl_module, input=input, print_log=False)
        except Exception as e:
            msglogger.critical("_do_summary failed")
            msglogger.critical(str(e))

        pl_module.train()
        return res


def prod(seq):
    """
    Args:
      seq:
    Returns:
    """
    result = 1.0
    for number in seq:
        result *= number
    return int(result)


def get_zero_op(node, output, args, kwargs):
    return 0, 0, ""


def get_conv(node, output, args, kwargs):
    volume_ofm = prod(output.shape)
    weight = args[1]
    out_channels = weight.shape[0]
    in_channels = weight.shape[1]
    kernel_size = weight.shape[2]
    num_weights = out_channels * in_channels / kwargs['groups'] * kernel_size**2
    macs = volume_ofm * in_channels / kwargs['groups'] * kernel_size
    attrs = "k=" + "(%d, %d)" % (kernel_size, kernel_size)
    attrs += ", s=" + "(%d, %d)" % (kwargs['stride'], kwargs['stride'])
    attrs += ", g=(%d)" % kwargs['groups']
    attrs += ", dsc=(%s)" % str(
        in_channels == out_channels == kwargs['groups']
    )
    attrs += ", d=" + "(%d, %d)" % (kwargs['dilation'], kwargs['dilation'])
    return num_weights, macs, attrs


def get_linear(node, output, args, kwargs):
    weight = args[1]
    in_features = weight.shape[0]
    out_features = weight.shape[1]
    num_weights = macs = in_features * out_features
    attrs = ""
    return num_weights, macs, attrs


def get_type(node):
    try:
        return node.name.split('_')[-2]
    except Exception as e:
        pass
    return node.name


class MACSummaryInterpreter(fx.Interpreter):
    def __init__(self, module: torch.nn.Module):
        tracer = GraphConversionTracer()
        traced_graph = tracer.trace(module)
        gm = fx.GraphModule(dict(module.params), traced_graph)
        super().__init__(gm)

        self.count_function = {
            conv2d: get_conv,
            linear: get_linear,
            add: get_zero_op,
        }

        self.data = {
            "Name": [],
            "Type": [],
            "Attrs": [],
            "IFM": [],
            "IFM volume": [],
            "OFM": [],
            "OFM volume": [],
            "Weights volume": [],
            "MACs": [],
        }

    def run_node(self, n : torch.fx.Node):
        try:
            out = super().run_node(n)
        except Exception as e:
            print(str(e))
        if n.op == 'call_function':
            try:
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                num_weights, macs, attrs = self.count_function.get(n.target, get_zero_op)(n, out, args, kwargs)
                self.data['Name'] += [n.name]
                self.data['Type'] += [get_type(n)]
                self.data['Attrs'] += [attrs]
                self.data['IFM'] += [tuple(args[0].shape)]
                self.data['IFM volume'] += [prod(args[0].shape)]
                self.data['OFM'] += [tuple(out.shape)]
                self.data['OFM volume'] += [prod(out.shape)]
                self.data['Weights volume'] += [int(num_weights)]
                self.data['MACs'] += [int(macs)]
            except Exception as e:
                msglogger.warning("Summary of node %s failed: %s", n.name, str(e))
        return out


class FxMACSummaryCallback(MacSummaryCallback):
    def _do_summary(self, pl_module, input=None, print_log=True):
        interpreter = MACSummaryInterpreter(pl_module.model)
        dummy_input = input
        if dummy_input is None:
            dummy_input = pl_module.example_feature_array
        dummy_input = dummy_input.to(pl_module.device)
        interpreter.run(dummy_input)

        total_macs = 0.0
        total_acts = 0.0
        total_weights = 0.0
        estimated_acts = 0.0

        try:
            df = pd.DataFrame(interpreter.data)
            t = tabulate(df, headers="keys", tablefmt="psql", floatfmt=".5f")
            total_macs = df["MACs"].sum()
            total_acts = df["IFM volume"][0] + df["OFM volume"].sum()
            total_weights = df["Weights volume"].sum()
            estimated_acts = 2 * max(df["IFM volume"].max(), df["OFM volume"].max())
            if print_log:
                msglogger.info("\n" + str(t))
                msglogger.info("Total MACs: " + "{:,}".format(total_macs))
                msglogger.info("Total Weights: " + "{:,}".format(total_weights))
                msglogger.info("Total Activations: " + "{:,}".format(total_acts))
                msglogger.info(
                    "Estimated Activations: " + "{:,}".format(estimated_acts)
                )
        except RuntimeError as e:
            msglogger.warning("Could not create performance summary: %s", str(e))
            return OrderedDict()

        res = OrderedDict()
        res["total_macs"] = total_macs
        res["total_weights"] = total_weights
        res["total_act"] = total_acts
        res["est_act"] = estimated_acts

        return res

