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

import pytorch_lightning as pl
import torch
from torch.ao.quantization import (
    get_default_qat_qconfig_mapping,
    get_default_qconfig_mapping,
)
from torch.ao.quantization.quantize_fx import convert, prepare_qat_fx

logger = logging.getLogger(__name__)


class QuantizationCallback(pl.Callback):
    def __init__(self, is_qat: bool = True, qconfig="x86"):
        super().__init__()

        self.qconfig = get_default_qat_qconfig_mapping(qconfig)

        self.is_qat = is_qat

        if not self.is_qat:
            logger.critical(
                "Quantization is currently not supported for non-qat training"
            )
            logger.critical("Falling back to qat training")

    def on_fit_start(self, trainer, pl_module):
        device = pl_module.device
        pl_module.cpu()
        pl_module.eval()
        pl_module.model = prepare_qat_fx(
            pl_module.model, self.qconfig, pl_module.example_feature_array
        )
        # model is any PyTorch model
        # pl_module.model.apply(torch.ao.quantization.disable_fake_quant)
        # pl_module.model.apply(torch.ao.quantization.disable_observer)
        pl_module.train()
        pl_module.to(device)

        logger.info("quantized module:\n %s", pl_module.model.print_readable(False))

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pass
        # FIXME: check if we should convert here
        # pl_module.model = convert(pl_module.model)
        # breakpoint()

    def on_on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        pass
