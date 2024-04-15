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
import json
import logging
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from .base import InferenceBackendBase, ProfilingResult
from .utils import symbolic_batch_dim

try:
    import onnx
    import onnxruntime
except ImportError:
    onnx = None
    onnxruntime = None


class OnnxruntimeBackend(InferenceBackendBase):
    """Inference Backend for tensorflow"""

    def __init__(
        self,
        repeat=10,
        warmup=2,
    ):
        self.repeat = repeat
        self.warmup = warmup

        self.tmp_pdir = None
        self.onnx_model = None

        self.sesssion = None

        if onnx is None or onnxruntime is None:
            raise Exception(
                "Could not find required libraries for onnxruntime backend please install with poetry instell -E onnxrt-backend"
            )

    def prepare(self, model):
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

        self.onnx_model = tmp_dir / "model_opt.onnx"
        self.session = onnxruntime.InferenceSession(self.onnx_model)

    def run(self, *inputs):
        return self._run(*inputs)

    def profile(self, *inputs):
        result = self._run(*inputs, profile=True)
        return result

    def _run(self, *inputs, profile=False):
        logging.info("running onnxruntime backend on batch")

        if profile:
            # Unfortunately, for profiling we need to recreate the session each time
            options = onnxruntime.SessionOptions()
            options.enable_profiling = True

            session = onnxruntime.InferenceSession(self.onnx_model, options)
        else:
            session = self.session

        # Get the input names
        input_names = [input.name for input in session.get_inputs()]

        inputs = {k: inp.numpy() for k, inp in zip(input_names, inputs)}

        # Run the model
        if profile:
            for i in range(self.warmup):
                result = session.run(None, inputs)

            start = time.perf_counter()
            for i in range(self.repeat):
                result = session.run(None, inputs)
            end = time.perf_counter()

            duration = (end - start) / self.repeat
        else:
            result = session.run(None, inputs)

        # Convert to torch tensors
        result = [torch.from_numpy(res) for res in result]
        result = result[0] if len(result) == 1 else result

        if profile:
            profile_file = session.end_profiling()
            with open(profile_file, "r") as f:
                profile = json.load(f)

            metrics = {"duration": duration}

            return ProfilingResult(outputs=result, metrics=metrics, profile=profile)

        return result

    @classmethod
    def available(cls):
        return onnx is not None and onnxruntime is not None
