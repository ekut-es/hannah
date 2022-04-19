import json
import os

import torch.nn as nn
from pytorch_lightning import Callback


class TestDumperCallback(Callback):
    def __init__(self, output_dir="."):
        self.output_dir = output_dir

    def on_test_start(self, pl_trainer, pl_model):
        print("Activating layer dumping")

        def dump_layers(model, output_dir):
            class DumpForwardHook:
                def __init__(self, module, output_dir):
                    self.module = module
                    self.output_dir = output_dir
                    try:
                        os.makedirs(self.output_dir)
                    except Exception:
                        pass

                    self.count = 0

                def __call__(self, module, input, output):

                    if self.count >= 100:
                        return

                    output_name = (
                        self.output_dir + "/output_" + str(self.count) + ".json"
                    )

                    output_copy = output.cpu().tolist()

                    with open(output_name, "w") as f:
                        f.write(json.dumps(output_copy))

                    self.count += 1

            for module_name, module in model.named_modules():

                if type(module) in [nn.ReLU, nn.Hardtanh]:

                    module.register_forward_hook(
                        DumpForwardHook(
                            module, output_dir + "/test_data/layers/" + module_name
                        )
                    )

                if type(module) in [nn.Conv1d]:
                    module.register_forward_hook(
                        DumpForwardHook(
                            module, output_dir + "/test_data/layers/" + module_name
                        )
                    )

        dump_layers(pl_model, self.output_dir + "/layer_outputs")
