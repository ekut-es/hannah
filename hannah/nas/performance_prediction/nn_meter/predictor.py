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
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import onnx
import torch

from hannah.backends.utils import symbolic_batch_dim

try:
    import nn_meter
except:
    nn_meter = None


class NNMeterPredictor:
    def __init__(self, hardware_name, predictor_version: Optional[float] = None):
        if nn_meter is None:
            raise ImportError("nn_meter is not installed")

        self._predictor = nn_meter.load_latency_predictor(
            hardware_name, predictor_version
        )

    def predict(self, model):
        self.tmp_dir = TemporaryDirectory()
        tmp_dir = Path(self.tmp_dir.name)

        logging.info("transfering model to onnx")
        dummy_input = model.example_input_array
        torch.onnx.export(model, dummy_input, tmp_dir / "model.onnx", verbose=False)
        logging.info("Creating onnxrt-model")
        onnx_model = onnx.load(tmp_dir / "model.onnx")
        symbolic_batch_dim(onnx_model)

        # Save the onnx model
        onnx.save(onnx_model, tmp_dir / "model_opt.onnx")

        latency = self._predictor.predict(str(tmp_dir / "model_opt.onnx"), "onnx")

        return {"duration": latency}
