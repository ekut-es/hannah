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
import collections
import copy
import io
import logging
import os
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import torch

try:
    from hannah.backends.base import (
        ClassifierModule,
        InferenceBackendBase,
        ProfilingResult,
    )
    from hannah.nas.export.onnx import to_onnx
except ImportError:
    # thats not a good thing to do, but works for now
    class ClassifierModule:
        def __init__(self, model):
            self.model = model

        @property
        def example_input_array(self):
            return self.model

    class InferenceBackendBase: ...

    ProfilingResult = namedtuple(
        "ProfilingResult", field_names=["outputs", "metrics", "profile"]
    )

    def to_onnx(lightning_model) -> "onnx.ModelProto":
        return lightning_model.onnx_model


try:
    from netdeployonnx.common.netclient_remote import (
        NetClient,
        get_netclient_from_connect,
    )
except ImportError as ie:
    logging.warning(f"Could not import netdeployonnx: {ie}")
    NetClient = None
    get_netclient_from_connect = None
try:
    import asyncio

    import grpc
    import onnx
except ImportError:
    grpc = None
    onnx = None


class GRPCBackend(InferenceBackendBase):
    def __init__(self, *args, client_connect: str = "localhost:28329", **kwargs):
        self.client_connect = client_connect
        self.auth: Path | str | bytes | None = kwargs.pop("auth", None)
        self.device_filter: list[dict] = kwargs.pop(
            "device_filter", [{"model": "EVKit_V1"}]
        )
        # make sure each is a dict
        self.device_filter = [dict(f) for f in self.device_filter]

        self.keepalive_timeout: float = kwargs.pop("keepalive_timeout", 4)
        self.function_timeout: float = kwargs.pop("function_timeout", 4)

        self.should_reraise: bool = kwargs.pop("should_reraise", False)

        self._client = None
        self.modelbytes = None
        super().__init__(*args, **kwargs)

    @property
    def client(self) -> NetClient:
        if self._client is None:
            try:
                # either it is a path
                if isinstance(self.auth, (str, Path)):
                    if os.path.exists(self.auth):
                        with open(self.auth, "rb") as f:
                            auth = f.read()
                    else:
                        raise FileNotFoundError(f"File {self.auth} not found")
                elif isinstance(self.auth, bytes):
                    auth = self.auth
                else:
                    auth = None
                self._client = get_netclient_from_connect(
                    self.client_connect,
                    auth,
                    keepalive_timeout=self.keepalive_timeout,
                )
            except Exception:
                raise  # ConnectionError(f"Could not connect to client: {ex}")
        return self._client

    def __del__(self):
        if self._client is not None:
            self._client.close()

    def prepare(self, module: ClassifierModule):
        """
        Prepare the model for execution on the target device

        Args:
          module: the classifier module to be exported

        """
        self.module = module
        quantized_model = copy.deepcopy(module.model)
        quantized_model.cpu()
        quantized_model.train(False)

        dummy_input = module.example_input_array.cpu()  # noqa: F841
        bytesio = io.BytesIO()

        def torch_aten_copy(g, input, *args):
            # print("TYPE=", type(input), "INPUT=", input)
            # if input is a torch.Value and is a float
            # then we can just return 2**input
            # exp_constant = None
            # exp_constant = 0
            # if exp_constant is not None:
            #     return g.op("Const", torch.tensor(2.0**exp_constant, dtype=float))
            return g.op("Identity", torch.tensor(2.0**1, dtype=float))

        torch.onnx.register_custom_op_symbolic("aten::copy", torch_aten_copy, 1)

        # torch.onnx.export(
        #     quantized_model,
        #     dummy_input,
        #     bytesio,
        #     verbose=False,
        #     opset_version=11,
        # )
        try:
            modelProto: onnx.ModelProto = to_onnx(quantized_model)  # noqa: N806
            bytesio.write(modelProto.SerializeToString())
        except Exception as ex:
            # exporting failed
            raise ex

        self.modelbytes = bytesio.getvalue()

    def run(self, *inputs) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """
        Run a batch on the target device

        Args:
          inputs: a list of torch tensors representing the inputs to be run on the target device, each tensor represents a whole batched input, so for models taking 1 parameter, the list will contain 1 tensor of shape (batch_size, *input_shape)

        Returns: the output(s) of the model as a torch tensor or a Sequence of torch tensors for models producing multiple outputs

        """  # noqa: E501
        return self._run(*inputs, profiling=False)

    def profile(self, *inputs: torch.Tensor) -> ProfilingResult:
        """Do a profiling run on the target device

        Args:
            inputs: a list of torch tensors representing the inputs to be run on the target device, each tensor represents a whole batched input, so for models taking 1 parameter, the list will contain 1 tensor of shape (batch_size, *input_shape)

        Returns: a ProfilingResult object containing the outputs of the model, the metrics obtained from the profiling run and the raw profile in a backend-specific format
        """  # noqa: E501
        return self._run(*inputs, profiling=True)

    async def _run_async(
        self, *inputs: torch.Tensor, profiling: bool = False
    ) -> Union[torch.Tensor, Sequence[torch.Tensor], ProfilingResult]:
        if get_netclient_from_connect is None:
            raise ImportError("netdeployonnx not installed")
        with get_netclient_from_connect(
            self.client_connect,
            self.auth,
            keepalive_timeout=self.keepalive_timeout,
            should_reraise=self.should_reraise,
        ) as client:
            async with client.connect(filters=self.device_filter) as device:
                input_bytes: bytes = b""
                result_dict = await device.deploy(
                    modelbytes=self.modelbytes,
                    input_bytes=input_bytes,
                    profiling=profiling,
                    timeout=self.function_timeout,
                )
                if profiling:
                    # FIXME: select the first? maybe average?
                    metrics = result_dict["metrics"]
                    if isinstance(result_dict, collections.abc.Iterable):
                        metrics = metrics[0] if len(metrics) > 0 else {}
                    return ProfilingResult(
                        outputs=result_dict["result"],
                        metrics=metrics,
                        profile=result_dict["deployment_execution_times"],
                    )
                else:
                    return result_dict["result"]
        raise ConnectionError("Could not connect to client")

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        return loop

    def _run(
        self, *inputs: torch.Tensor, profiling: bool = False
    ) -> Union[torch.Tensor, Sequence[torch.Tensor], ProfilingResult]:
        try:
            loop = self._get_loop()
            async_result = loop.run_until_complete(
                self._run_async(*inputs, profiling=profiling)
            )
            return async_result
        except TimeoutError:
            raise
        except ValueError:
            raise
        except ConnectionError:
            raise
        except grpc._channel._InactiveRpcError as irpce:
            raise Exception(
                "Inactive RPC Server (either server down or not reachable)"
            ) from irpce  # noqa: E501
        except Exception as ex:
            raise ex  # reraise

    @classmethod
    def available(cls) -> bool:
        """
        Check if the backend is available

        Returns: True if the backend is available, False otherwise

        """
        try:
            # TODO: check server availability?
            return (
                grpc is not None
                and onnx is not None
                and NetClient is not None
                and get_netclient_from_connect is not None
            )
        except Exception:
            pass
        return False
