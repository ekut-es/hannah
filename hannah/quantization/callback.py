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
from typing import Union

import pytorch_lightning as pl
import torch
from torch.ao.quantization import (
    QConfigMapping,
    get_default_qat_qconfig_mapping,
    get_default_qconfig_mapping,
)
from torch.ao.quantization.backend_config import (
    BackendConfig,
    get_executorch_backend_config,
    get_fbgemm_backend_config,
    get_native_backend_config,
    get_onednn_backend_config,
    get_qnnpack_backend_config,
    get_tensorrt_backend_config,
)
from torch.ao.quantization.quantize_fx import convert, prepare_qat_fx

logger = logging.getLogger(__name__)

config_mapping = {
    "fbgemm": get_fbgemm_backend_config,
    "native": get_native_backend_config,
    "qnnpack": get_qnnpack_backend_config,
    "tensorrt": get_tensorrt_backend_config,
    "excutorch": get_executorch_backend_config,
    "onednn": get_onednn_backend_config,
}


class QuantizationCallback(pl.Callback):
    def __init__(
        self,
        is_qat: bool = True,
        qconfig_mapping: Union[str, QConfigMapping] = "fbgemm",
        backend_config: Union[str, BackendConfig] = "native",
    ):
        super().__init__()

        if isinstance(backend_config, str):
            self.backend_config = config_mapping[backend_config]()
        else:
            self.backend_config = backend_config

        if isinstance(qconfig_mapping, str):
            if is_qat:
                self.qconfig_mapping = get_default_qat_qconfig_mapping(qconfig_mapping)
            else:
                self.qconfig_mapping = get_default_qconfig_mapping(qconfig_mapping)

        self.is_qat = is_qat

        if self.is_qat:
            logger.critical(
                "Quantization aware training will most likely not work with current tvm backend."
            )
            logger.critical("Falling back to qat training")

    def on_fit_start(self, trainer, pl_module):
        if self.is_qat:
            device = pl_module.device
            pl_module.cpu()
            pl_module.eval()
            pl_module.model = prepare_qat_fx(
                pl_module.model,
                self.qconfig_mapping,
                pl_module.example_feature_array,
                backend_config=self.backend_config,
            )
            # model is any PyTorch model
            # pl_module.model.apply(torch.ao.quantization.disable_fake_quant)
            # pl_module.model.apply(torch.ao.quantization.disable_observer)
            pl_module.train()
            pl_module.to(device)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pass
        # FIXME: check if we should convert here
        # pl_module.model = convert(pl_module.model)
        # breakpoint()

    def on_on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        pass
