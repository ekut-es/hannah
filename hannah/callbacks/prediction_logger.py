#
# Copyright (c) 2024 Hannah contributors.
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
from pytorch_lightning import Callback
from torch import is_tensor


class PredictionLogger(Callback):
    def setup(self, trainer, pl_module, stage):
        self.val_values = []
        self.test_values = []

    def on_validation_epoch_end(self, trainer, pl_module):
        self._write_values("val", self.test_values)

        self.val_values = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._log("val", self.val_values, batch, outputs, batch_idx, dataloader_idx)

    def on_test_epoch_end(self, trainer, pl_module):
        self._write_values("test", self.test_values)
        self.test_values = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._log("test", self.test_values, batch, outputs, batch_idx, dataloader_idx)

    def _log(self, stage, values, batch, outputs, batch_idx, dataloader_idx):
        values.append(outputs)

        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if is_tensor(value):
                    value = value.detach().cpu().numpy()

    def _write_values(self, stage, values):
        pass
