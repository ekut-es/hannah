import logging
from collections import OrderedDict

import pandas as pd
import torch
from speech_recognition.models.sinc import SincNet

from pytorch_lightning.callbacks import Callback
from tabulate import tabulate

msglogger = logging.getLogger()


class MacSummaryCallback(Callback):
    def walk_model(self, model, dummy_input):
        """Adapted from IntelLabs Distiller"""

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
            result = 1.0
            for number in seq:
                result *= number
            return int(result)

        def get_name_by_module(m):
            for module_name, mod in model.named_modules():
                if m == mod:
                    return module_name

        def collect(module, input, output):
            volume_ifm = prod(input[0].size())
            volume_ofm = prod(output.size())
            extra = get_extra(module, volume_ofm)
            if extra is not None:
                weights, macs, attrs = extra
            else:
                return
            data["Name"] += [get_name_by_module(module)]
            data["Type"] += [module.__class__.__name__]
            data["Attrs"] += [attrs]
            data["IFM"] += [tuple(input[0].size())]
            data["IFM volume"] += [volume_ifm]
            data["OFM"] += [tuple(output.size())]
            data["OFM volume"] += [volume_ofm]
            data["Weights volume"] += [int(weights)]
            data["MACs"] += [int(macs)]

        def get_extra(module, volume_ofm):
            classes = {
                torch.nn.Conv1d: get_conv,
                torch.nn.Conv2d: get_conv,
                SincNet: get_sinc_conv,
                torch.nn.Linear: get_fc,
            }

            for _class, method in classes.items():
                if isinstance(module, _class):
                    return method(module, volume_ofm)

            return get_generic(module)

        def get_conv_macs(module, volume_ofm):
            return volume_ofm * (
                module.in_channels / module.groups * prod(module.kernel_size)
            )

        def get_conv_attrs(module):
            attrs = (
                "k=" + "(" + (", ").join(["%d" % v for v in module.kernel_size]) + ")"
            )
            attrs += ", s=" + "(" + (", ").join(["%d" % v for v in module.stride]) + ")"
            attrs += ", g=%d" % module.groups
            attrs += ", d=" + "(" + ", ".join(["%d" % v for v in module.dilation]) + ")"
            return attrs

        def get_conv(module, volume_ofm):
            weights = (
                module.out_channels
                * module.in_channels
                / module.groups
                * prod(module.kernel_size)
            )
            macs = get_conv_macs(module, volume_ofm)
            attrs = get_conv_attrs(module)
            return weights, macs, attrs

        def get_sinc_conv(module, volume_ofm):
            weights = 2 * module.out_channels * module.in_channels / module.groups
            macs = get_conv_macs(module, volume_ofm)
            attrs = get_conv_attrs(module)
            return weights, macs, attrs

        def get_fc(module, volume_ofm):
            weights = macs = module.in_features * module.out_features
            attrs = ""
            return weights, macs, attrs

        def get_generic(module):
            if isinstance(module, torch.nn.Dropout):
                return
            weights = macs = 0
            attrs = ""
            return weights, macs, attrs

        hooks = list()

        for name, module in model.named_modules():
            if module != model:
                hooks += [module.register_forward_hook(collect)]

        _ = model(dummy_input)

        for hook in hooks:
            hook.remove()

        df = pd.DataFrame(data=data)
        return df

    def _do_summary(self, pl_module, print=True):
        dummy_input = pl_module.example_feature_array

        total_macs = 0.0
        total_acts = 0.0
        total_weights = 0.0
        estimated_acts = 0.0

        try:
            df = self.walk_model(pl_module.model, dummy_input)
            t = tabulate(df, headers="keys", tablefmt="psql", floatfmt=".5f")
            total_macs = df["MACs"].sum()
            total_acts = df["IFM volume"][0] + df["OFM volume"].sum()
            total_weights = df["Weights volume"].sum()
            estimated_acts = 2 * max(df["IFM volume"].max(), df["OFM volume"].max())
            if print:
                msglogger.info("\n" + str(t))
                msglogger.info("Total MACs: " + "{:,}".format(total_macs))
                msglogger.info("Total Weights: " + "{:,}".format(total_weights))
                msglogger.info("Total Activations: " + "{:,}".format(total_acts))
                msglogger.info(
                    "Estimated Activations: " + "{:,}".format(estimated_acts)
                )
        except RuntimeError as e:
            if print:
                msglogger.warning("Could not create performance summary: %s", str(e))
            return OrderedDict()

        res = OrderedDict()
        res["total_macs"] = total_macs
        res["total_weights"] = total_weights
        res["total_act"] = total_acts
        res["est_act"] = estimated_acts

        return res

    def on_train_start(self, trainer, pl_module):
        pl_module.eval()
        self._do_summary(pl_module)
        pl_module.train()

    def on_validation_epoch_end(self, trainer, pl_module):
        res = self._do_summary(pl_module, print=False)
        for k, v in res.items():
            pl_module.log(k, v)
