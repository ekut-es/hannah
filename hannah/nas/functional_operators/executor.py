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
from collections import defaultdict
from copy import deepcopy
from typing import Iterator, Tuple
import math
import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from .op import ChoiceOp, Op, Tensor, get_nodes


class BasicExecutor(torch.nn.Module):
    def __init__(self, net, input_node_name="input", init=None) -> None:
        super().__init__()
        self.output = deepcopy(net)
        self.input_node_name = input_node_name
        self.params = {}
        self.input = None
        self.input_data = None
        if init is not None:
            self.init = init
        else:
            self.init = torch.nn.init.kaiming_uniform_
        self.nodes = []

    def initialize(self):
        self.find_execution_order()
        self.register_modules()

    def register_modules(self) -> None:
        for node in self.nodes:
            op = self.node_dict[node]
            if isinstance(op, torch.nn.Module):
                self.add_module(node.replace(".", "_"), op)

    def initialize_tensor(self, node):
        if isinstance(node, Tensor):
            node_name = node.id.replace(".", "_")
            if node.grad:
                if node.name == "bias":
                    # get weight data
                    weight_name = node_name.replace("bias", "weight")
                    weight_param = self.get_parameter(weight_name)
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                        weight_param.data
                    )
                    # register bias
                    if fan_in != 0:
                        bound = 1 / math.sqrt(fan_in)
                        data = torch.empty(node.current_shape())
                        data = torch.nn.Parameter(
                            torch.nn.init.uniform_(data, -bound, bound)
                        )
                        self.register_parameter(node_name, data)
                else:  # weight tensor
                    data = torch.empty(node.current_shape())
                    data = torch.nn.Parameter(self.init(data, a=math.sqrt(5)))
                    self.register_parameter(node_name, data)
            if node.name == self.input_node_name:
                self.input = node
            if node.name == "running_mean":
                data = torch.zeros(node.current_shape())
                self.register_buffer(node_name, data)
            if node.name == "running_std":
                data = torch.ones(node.current_shape())
                self.register_buffer(node_name, data)
            node.executor = self

    def get_data(self, id):
        if id == "input":
            return self.input_data
        else:
            name = id.replace(".", "_")
            return getattr(self, name)

    # def parameters(self):
    #     queue = [self.output]
    #     visited = [self.output.id]

    #     while queue:
    #         node = queue.pop(0)
    #         if isinstance(node, Tensor) and isinstance(node.data, Parameter):
    #             yield node.data

    #         for operand in node.operands:
    #             while isinstance(operand, ChoiceOp):
    #                 active_op_index = operand.switch.evaluate()
    #                 operand = operand.operands[active_op_index]

    #             if operand.id not in visited:
    #                 queue.append(operand)
    #                 visited.append(operand.id)

    # TODO: Let parameters only return the active parameters
    def _all_parameters(self):
        for node in get_nodes(self.output):
            if isinstance(node, Tensor) and isinstance(node.data, Parameter):
                yield node.data

    # def check_operands_available(self, node, visited):
    #     for operand in node.operands:
    #         if operand.id not in visited:
    #             return False
    #     return True

    # def check_users_available(self, node, nodes):
    #     for user in node.users:
    #         if user not in nodes:
    #             return False
    #     return True

    def find_dependencies(self):
        nodes = []
        queue = [self.output]
        visited = [self.output.id]
        dependency_dict = {}
        self.node_dict = {}

        while queue:
            node = queue.pop(0)
            self.initialize_tensor(node)
            self.node_dict[node.id] = node
            dependency_dict[node.id] = []

            for operand in node.operands:
                while isinstance(operand, ChoiceOp):
                    active_op_index = operand.switch.evaluate()
                    operand = operand.operands[active_op_index]
                if operand.id not in dependency_dict[node.id]:
                    dependency_dict[node.id].append(operand.id)
                if operand.id not in visited:
                    queue.append(operand)
                    visited.append(operand.id)
        return dependency_dict

    def remove_from_dependency_dict(self, dependency_dict, node):
        new_dict = deepcopy(dependency_dict)
        for k, v in dependency_dict.items():
            if node in v:
                new_dict[k].remove(node)
        return new_dict

    def find_execution_order(self):
        dependency_dict = self.find_dependencies()
        self.forward_dict = deepcopy(dependency_dict)

        nodes = []
        queue = list(dependency_dict.keys())
        while queue:
            node = queue.pop(-1)
            if len(dependency_dict[node]) > 0:
                queue = [node] + queue
            else:
                nodes.append(node)
                dependency_dict = self.remove_from_dependency_dict(
                    dependency_dict, node
                )
        self.nodes = nodes
        self.nodes.remove("input")

    def find_active_modules(self):
        pass

    def forward(self, x):
        out = {"input": x}
        for node in self.nodes:
            operands = [out[n] for n in self.forward_dict[node]]
            out[node] = self.node_dict[node].forward(*operands)
        return out[node]

    def parametrization(self, flatten=True):
        return self.output.parametrization(flatten=flatten)

    def train(self, mode=True):
        if not mode:
            self.eval()
        else:
            for node in get_nodes(self.output):
                if isinstance(node, Op):
                    node.train()
            self.training = True

    def eval(self, mode=True):
        if not mode:
            self.train()
        else:
            for node in get_nodes(self.output):
                if isinstance(node, Op):
                    node.eval()
            self.training = False


class WeightSharingExecutor:
    def __init__(self) -> None:
        pass
