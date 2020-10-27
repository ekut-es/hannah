import logging
from collections import OrderedDict

import distiller

from pytorch_lightning.callbacks import Callback
from tabulate import tabulate

msglogger = logging.getLogger()


class MacSummaryCallback(Callback):
    def _do_summary(self, pl_module, print=True):
        dummy_input = pl_module.example_input_array

        total_macs = 0.0
        total_acts = 0.0
        total_weights = 0.0
        estimated_acts = 0.0

        try:
            df = distiller.model_performance_summary(
                pl_module.model, dummy_input, dummy_input.shape[0]
            )
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
        self._do_summary(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        res = self._do_summary(pl_module, print=False)
        trainer.logger.log_metrics(res)
