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
from typing import Any, Dict, Tuple

import torch.fx
from torch.fx.node import Argument, Node, Target

from hannah.nas.functional_operators.executor import BasicExecutor
from hannah.nas.functional_operators.operators import conv2d, relu


class SearchSpaceTracer(torch.fx.Tracer):
    def create_node(
        self,
        kind: str,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name=None,
        type_expr=None,
    ) -> Node:
        if kind == "call_function" and "id" in kwargs:
            name = kwargs["id"]
        return super().create_node(kind, target, args, kwargs, name, type_expr)

    def create_arg(self, a: Any) -> Argument:
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node("get_attr", n, (), {})
            raise NameError("parameter is not a member of this module")
        if isinstance(a, torch.Tensor):
            if isinstance(self.root, BasicExecutor):
                if isinstance(a, torch.nn.Parameter):
                    for n, p in self.root.named_parameters():
                        if a is p:
                            return self.create_node("get_attr", n, (), {})
                    raise NameError("parameter is not a member of this module")
                elif isinstance(a, torch.Tensor):
                    for n_, p_ in self.root.named_buffers():
                        if a is p_:
                            return self.create_node("get_attr", n_, (), {})
        return super().create_arg(a)


class InliningTracer(SearchSpaceTracer):
    """Inline all search space functions, into the graph.

    This generates a standard pytorch.fx graph module containing only replacing the search space parametrizable functions with their equivalent form torch.functional`
    """

    FNS_TO_INLINE = [conv2d, relu]

    def create_node(
        self,
        kind: str,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name=None,
        type_expr=None,
    ) -> Node:
        if kind == "call_function" and "id" in kwargs:
            name = kwargs["id"]

        if kind == "call_function" and target in self.FNS_TO_INLINE:
            tracer = torch.fx.proxy.GraphAppendingTracer(self.graph)

            proxy_args = torch.fx.node.map_arg(
                args, lambda x: torch.fx.Proxy(x, tracer)
            )
            proxy_kwargs = torch.fx.node.map_arg(
                kwargs, lambda x: torch.fx.Proxy(x, tracer)
            )

            return target(*proxy_args, **proxy_kwargs).node

        return super().create_node(kind, target, args, kwargs, name, type_expr)
