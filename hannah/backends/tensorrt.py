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
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

try:
    import tensorrt as trt
    from cuda import cuda, cudart
except ModuleNotFoundError:
    trt = None
    cuda = None
    cudart = None

from .base import InferenceBackendBase, ProfilingResult


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
    )


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
    )


def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class TensorRTBackend(InferenceBackendBase):
    def __init__(
        self, val_batches=1, test_batches=1, val_frequency=10, warmup=10, repeat=30
    ):
        super().__init__(
            val_batches=val_batches,
            test_batches=test_batches,
            val_frequency=val_frequency,
        )

        if trt is None or cuda is None or cudart is None:
            raise RuntimeError(
                "TensorRT is not available, please install with tensorrt extra activated."
            )

        self.trt_logger = trt.Logger(trt.Logger.INFO)

        self.builder = trt.Builder(self.trt_logger)

        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = 8 * (2**30)  # 8 GB

        self.batch_size = None
        self.network = None
        self.parser = None

        self.engine = None
        self.context = None

    def output_spec(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def prepare(self, module):
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            onnx_path = tmp_dir / "model.onnx"

            logging.info("transfering model to onnx")
            dummy_input = module.example_input_array
            dummy_input = dummy_input.to(module.device)
            torch.onnx.export(module, dummy_input, onnx_path, verbose=False)

            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

            self.network = self.builder.create_network(network_flags)
            self.parser = trt.OnnxParser(self.network, self.trt_logger)
            onnx_path = os.path.realpath(onnx_path)
            with open(onnx_path, "rb") as f:
                if not self.parser.parse(f.read()):
                    logging.error("Failed to load ONNX file: {}".format(onnx_path))
                    for error in range(self.parser.num_errors):
                        logging.error(self.parser.get_error(error))

            self.engine = self.builder.build_engine(self.network, self.config)
            self.context = self.engine.create_execution_context()

            # Setup I/O bindings
            self.inputs = []
            self.outputs = []
            self.allocations = []
            for i in range(self.engine.num_bindings):
                is_input = False
                if self.engine.binding_is_input(i):
                    is_input = True
                name = self.engine.get_binding_name(i)
                dtype = self.engine.get_binding_dtype(i)
                shape = self.engine.get_binding_shape(i)
                if is_input:
                    self.batch_size = shape[0]
                size = np.dtype(trt.nptype(dtype)).itemsize
                for s in shape:
                    size *= s

                if size <= 0:
                    continue

                print("Allocation", name, "size: ", size)
                allocation = cuda_call(cudart.cudaMalloc(size))
                binding = {
                    "index": i,
                    "name": name,
                    "dtype": np.dtype(trt.nptype(dtype)),
                    "shape": list(shape),
                    "allocation": allocation,
                }
                self.allocations.append(allocation)
                if self.engine.binding_is_input(i):
                    self.inputs.append(binding)
                else:
                    self.outputs.append(binding)

            assert self.batch_size > 0
            assert len(self.inputs) > 0
            assert len(self.outputs) > 0
            assert len(self.allocations) > 0

    def run(self, *inputs):
        output = np.zeros(*self.output_spec())

        memcpy_host_to_device(
            self.inputs[0]["allocation"], np.ascontiguousarray(inputs[0].cpu().numpy())
        )
        self.context.execute_v2(self.allocations)
        memcpy_device_to_host(output, self.outputs[0]["allocation"])

        result = torch.from_numpy(output)

        return result

    def profile(self, *inputs):
        output = np.zeros(*self.output_spec())

        memcpy_host_to_device(
            self.inputs[0]["allocation"], np.ascontiguousarray(inputs[0].cpu().numpy())
        )

        for _ in range(self.warmup):
            self.context.execute_v2(self.allocations)

        start = time.perf_counter()
        for _ in range(self.repeat):
            self.context.execute_v2(self.allocations)
        end = time.perf_counter()

        duration = (end - start) / self.repeat

        memcpy_device_to_host(output, self.outputs[0]["allocation"])

        result = torch.from_numpy(output)

        return ProfilingResult(
            outputs=result, metrics={"duration": duration}, profile=None
        )

    @classmethod
    def available(cls):
        if trt is not None and cuda is not None and cudart is not None:
            return cuda.cuDeviceGetCount()[1] > 0

        return False
