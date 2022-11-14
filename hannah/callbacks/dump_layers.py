#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
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
import json
import logging
import os

import torch.nn as nn
from pytorch_lightning import Callback

import hannah.models.factory.qat as qat

logger = logging.getLogger(__name__)


class TestDumperCallback(Callback):
    def __init__(self, output_dir="."):
        self.output_dir = output_dir

    def on_test_start(self, pl_trainer, pl_model):
        logger.info("Activating layer dumping")

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

                if type(module) in [
                    nn.Conv1d,
                    qat.Conv1d,
                    qat.ConvBn1d,
                    qat.ConvBnReLU1d,
                    qat.ConvReLU1d,
                    qat.Linear,
                ]:
                    module.register_forward_hook(
                        DumpForwardHook(
                            module, output_dir + "/test_data/layers/" + module_name
                        )
                    )

        dump_layers(pl_model, self.output_dir + "/layer_outputs")
