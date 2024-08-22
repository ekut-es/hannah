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
import time

from .base import InferenceBackendBase, ProfilingResult


class TorchMobileBackend(InferenceBackendBase):
    """Inference backend for torch mobile"""

    def __init__(self, warmup=2, repeat=10):
        self.warmup = warmup
        self.repeat = repeat
        self.script_module = None

    def prepare(self, model):
        logging.info("Preparing model for target")
        self.script_module = model.to_torchscript(method="trace")

    def run(self, *inputs):
        if inputs is None or len(inputs) == 0:
            logging.critical("Backend batch is empty")
            return None

        return self.script_module(*inputs)

    def profile(self, *inputs):
        if inputs is None or len(inputs) == 0:
            logging.critical("Backend batch is empty")
            return None

        for i in range(self.warmup):
            self.script_module(*inputs)

        start = time.perf_counter()
        for i in range(self.repeat):
            results = self.script_module(*inputs)
        end = time.perf_counter()

        duration = (end - start) / self.repeat

        return ProfilingResult(
            outputs=results,
            metrics={"duration": duration},
            profile=None,
        )

    @classmethod
    def available(cls):
        return True
