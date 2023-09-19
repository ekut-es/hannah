from copy import deepcopy
from typing import Iterator, Tuple
import torch
from hannah.nas.functional_operators.op import ChoiceOp, Op, Tensor, get_nodes
from collections import defaultdict
from torch.nn.parameter import Parameter


class BasicExecutor(torch.nn.Module):
    def __init__(self, net, input_node_name='input', init=None) -> None:
        super().__init__()
        self.output = deepcopy(net)
        self.input_node_name = input_node_name
        # self.param_list = torch.nn.ParameterList()
        # self.param_dict = torch.nn.ParameterDict()
        self.params = {}
        self.input = None
        if init is not None:
            self.init = init
        else:
            self.init = torch.nn.init.xavier_uniform_
        self.nodes = []

    def initialize(self):
        # FIXME: Only initialize used tensors
        for node in get_nodes(self.output):
            if isinstance(node, Tensor):
                if node.grad:
                    data = torch.empty(node.current_shape())
                    data = self.init(data)
                    node.feed_data(data, grad=True)
                    node.executor = self
                    # self.param_list.append(node.data)
                    node_name = node.id.replace(".", "_")
                    # self.param_dict[node_name] = node.data
                if node.name == self.input_node_name:
                    self.input = node
                    self.input.executor = self
                if node.name == 'running_mean':
                    data = torch.zeros(node.current_shape())
                    node.feed_data(data, grad=False)
                if node.name == 'running_std':
                    data = torch.ones(node.current_shape())
                    node.feed_data(data, grad=False)
                # self.module_list.append(node)
                node.executor = self
                self.params[node.id] = node.data

        self.find_execution_order()

    def parameters(self):
        queue = [self.output]
        visited = [self.output.id]

        while queue:
            node = queue.pop(0)
            if isinstance(node, Tensor) and isinstance(node.data, Parameter):
                yield node.data

            for operand in node.operands:
                while isinstance(operand, ChoiceOp):
                    active_op_index = operand.switch.evaluate()
                    operand = operand.operands[active_op_index]

                if operand.id not in visited:
                    queue.append(operand)
                    visited.append(operand.id)

    def _all_parameters(self):
        for node in get_nodes(self.output):
            if isinstance(node, Tensor) and isinstance(node.data, Parameter):
                yield node.data

    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        for node in get_nodes(self.output):
            if isinstance(node, Tensor) and isinstance(node.data, Parameter):
                yield (node.id, node.data)

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
                dependency_dict = self.remove_from_dependency_dict(dependency_dict, node)
        self.nodes = nodes

    def find_active_modules(self):
        pass

    def forward(self, x):
        # self.input.data = x
        self.params[self.input.id] = x
        out = {}
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

    def eval(self, mode=True):
        if not mode:
            self.train()
        for node in get_nodes(self.output):
            if isinstance(node, Op):
                node.eval()


class WeightSharingExecutor:
    def __init__(self) -> None:
        pass
