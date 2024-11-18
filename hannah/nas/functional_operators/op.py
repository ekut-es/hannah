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
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import wraps
from typing import Any

import torch

from hannah.nas.core.expression import Expression
from hannah.nas.core.parametrized import is_parametrized
from hannah.nas.dataflow.data_type import FloatType
from hannah.nas.expressions.choice import Choice
from hannah.nas.expressions.utils import extract_parameter_from_expression
from hannah.nas.functional_operators.lazy import lazy
from hannah.nas.parameters.parameters import (
    CategoricalParameter,
    IntScalarParameter,
    Parameter,
)
from hannah.nas.parameters.parametrize import parametrize


def nodes_in_scope(node, inputs):
    queue = [node]
    visited = [node]

    while queue:
        n = queue.pop(-1)
        yield n
        for o in n.operands:
            # Cant use "in" because of EQ-Condition
            if not any([o is i for i in inputs]) and o not in visited:
                queue.append(o)
                visited.append(o)


def get_nodes(node):
    queue = [node]
    visited = [node]

    while queue:
        n = queue.pop(-1)
        yield n
        for o in n.operands:
            # Cant use "in" because of EQ-Condition
            if o not in visited:
                queue.append(o)
                visited.append(o)


_id = 0


def get_unique_id():
    global _id
    _id += 1
    return _id


def get_highest_scope_counter(scope, scope_dict):
    if scope in scope_dict:
        scope_dict[scope] += 1
    else:
        scope_dict[scope] = 0
    return scope_dict[scope]


def scope(function):
    """Decorator defining a scope in a search space. The id of every subcomponent (operators or lower-hierarchy scopes)
    enclosed in a function decorated with this will be prefixed with the name of the function, creating a
    hierarchical scope.
    """

    @wraps(function)
    def set_scope(*args, **kwargs):
        assert (
            "global_scope_stack" in globals()
        ), "No scope tracking found, did you wrap the search space with @search_space?"

        inputs = [a for a in args if isinstance(a, (Op, Tensor))] + [
            a for k, a in kwargs.items() if isinstance(a, (Op, Tensor))
        ]
        name = function.__name__
        global global_scope_stack
        ct = get_highest_scope_counter(name, global_scope_stack[-1])
        global_scope_stack.append({})
        out = function(*args, **kwargs)
        for n in nodes_in_scope(out, inputs):
            n.setid(f"{name}_{ct}.{n.id}")
        global_scope_stack.pop()
        return out

    return set_scope


def search_space(function):
    """Decorator to define a search space. For correct scoping,
    a search space containing functional ops must be enclosed by
    a function decorated with @search_space.
    """

    @wraps(function)
    def search_space_limits(*args, **kwargs):
        global global_scope_stack
        global_scope_stack = [{}]
        out = scope(function)(*args, **kwargs)
        del global_scope_stack
        return out

    return search_space_limits


@parametrize
class Op(torch.nn.Module):
    def __init__(self, name, *args, **kwargs) -> None:
        super().__init__()
        self.operands = []
        self.users = []
        self.name = name
        self.id = name
        self._shape = None
        self._train = True

    def __call__(self, *operands) -> Any:
        new_op = self  # FIXME: Just use self?
        for operand in operands:
            operand.connect(new_op)

        # Some Ops (ChoiceOp) can be called multiple times and already have a counter
        if not len(self.id.split(".")[-1].split("_")) > 1:
            assert (
                "global_scope_stack" in globals()
            ), "No scope tracking found, did you wrap the search space with @search_space?"
            global global_scope_stack
            ct = get_highest_scope_counter(self.name, global_scope_stack[-1])
            self.id = f"{self.id}_{ct}"
            # self.setid(f"{self.id}_{ct}")
        return new_op

    def connect(self, node):
        if is_parametrized(self):
            param_keys = list(node._PARAMETERS.keys())
            param_keys = [k for k in param_keys if self.name in k]
            if param_keys:
                current_count = max([int(k.split("_")[-1]) for k in param_keys]) + 1
            else:
                current_count = 0
            node._PARAMETERS[f"{self.name}_{current_count}"] = self

        node.operands.append(self)
        self.users.append(node)

    def forward(self, *operands):
        return self._forward_implementation(*operands)

    def setid(self, new_id):
        to_remove = []
        new_params = {}
        for name, par in self._PARAMETERS.items():
            if isinstance(par, Parameter):
                new_params[f"{new_id}.{par.name}"] = par
                to_remove.append(name)
                par.id = f"{new_id}.{par.name}"
        for name in to_remove:
            self._PARAMETERS.pop(name)
        for name, par in new_params.items():
            self._PARAMETERS[name] = par

        self.id = new_id

    def shape(self):
        if self._shape is None:
            self._shape = self.shape_fun()
        return self._shape

    @abstractmethod
    def _forward_implementation(self, *operands): ...

    def shape_fun(self):
        raise NotImplementedError

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def __repr__(self) -> str:
        return str(self.__class__).split(".")[-1].split("'")[0] + f"({self.id})"

    def __deepcopy__(self, memo):
        new_op = self.__class__.__new__(self.__class__)

        memo[id(self)] = new_op

        # print(self.id, id(self))
        # print(f"{self.__class__.__name__}")
        for k, v in self.__dict__.items():
            # print("   ", k)

            setattr(new_op, k, deepcopy(v, memo))

        return new_op


@torch.fx.wrap
def get_tensor_data(tensor):
    return tensor.data


@parametrize
class Tensor:
    def __init__(self, name, shape, axis, dtype=FloatType(), grad=False) -> None:
        super().__init__()
        self.name = name
        self.id = name
        self._shape = shape
        self.dtype = dtype

        # FIXME: Maybe check for lists/tuples in @parametrize?
        # FIXME: What if a parameter is defined elsewhere (e.g. conv) --> not good
        # for s in self._shape:
        #     if is_parametrized(s):
        #         # FIXME: IDs of parameters
        #         if s.id is None:  # Else: parameter is registered elsewhere
        #             s.id = self.id + '.' + s.name
        #             self._PARAMETERS[s.id] = s
        self.axis = axis
        self.users = []
        self.operands = []
        self.data = None
        self.grad = grad
        self.executor = None

    def forward(self, *operands):
        # TODO: Shape checking
        return self.executor.get_data(self.id)

    def _forward_implementation(self, *operands):
        return self.forward()

    def connect(self, node):
        if is_parametrized(self):
            node._PARAMETERS[self.name] = self
        node.operands.append(self)
        self.users.append(node)

    def setid(self, new_id):
        to_remove = []
        new_params = {}
        for name, par in self._PARAMETERS.items():
            new_params[f"{new_id}.{par.name}"] = par
            to_remove.append(name)
            par.id = f"{new_id}.{par.name}"
        for name in to_remove:
            self._PARAMETERS.pop(name)
        for name, par in new_params.items():
            self._PARAMETERS[name] = par
        self.id = new_id

    def shape(self):
        return self._shape

    def current_shape(self):  # FIXME: better naming
        return tuple([lazy(s) for s in self._shape])

    # def feed_data(self, data, grad=False):
    #     self.data = data

    def __fx_create_arg__(self, tracer: torch.fx.Tracer):
        return tracer.create_node(
            "call_function",
            Tensor,
            args=(
                # tracer.create_arg(self.data),
            ),
            kwargs={},
            name=self.id,
        )

    def __repr__(self):
        return f"Tensor({self.id})"


# @torch.fx.wrap
# def choice_forward()


@parametrize
class ChoiceOp(Op):
    def __init__(self, *options, switch=None) -> None:
        super().__init__(name="ChoiceOp")
        self.options = list(options)
        self.operands = self.options
        self.called = False  # FIXME: better name

        if switch is not None:
            self.switch = switch
        else:
            self.switch = self.add_param(
                "choice",
                IntScalarParameter(min=0, max=len(self.options) - 1, name="choice"),
            )

    def __call__(self, *operands):
        if self.called:
            return super().__call__(*operands)
        else:
            self._connect_options(*operands)
            self.called = True
            return self

    def _connect_options(self, *operands):
        for i, node_opt in enumerate(self.options):
            self.options[i] = node_opt(*operands)
            if is_parametrized(self.options[i]):
                self._PARAMETERS[self.options[i].id] = self.options[i]  # FIXME:
        assert (
            "global_scope_stack" in globals()
        ), "No scope tracking found, did you wrap the search space with @search_space?"
        global global_scope_stack
        ct = get_highest_scope_counter(self.name, global_scope_stack[-1])
        self.id = f"{self.id}_{ct}"

    def shape_fun(self):
        shapes = []
        for node_opt in self.options:
            shapes.append(node_opt.shape())
        shape_choice = Choice(shapes, choice=self.switch)
        return shape_choice

    def _forward_implementation(self, *operands):
        c = self.switch.evaluate()
        out = self.options[c]._forward_implementation(*operands)
        return out

    def setid(self, new_id):
        to_remove = []
        new_params = {}
        for name, par in self._PARAMETERS.items():
            if isinstance(par, Parameter):
                new_params[f"{new_id}.{par.name}"] = par
                to_remove.append(name)
                par.id = f"{new_id}.{par.name}"
        for name in to_remove:
            self._PARAMETERS.pop(name)
        for name, par in new_params.items():
            self._PARAMETERS[name] = par

        self.id = new_id


class OptionalOp(ChoiceOp):
    def __init__(self, node, switch=None) -> None:
        # optional -> only two choices
        switch = IntScalarParameter(min=0, max=1, name="choice")
        super().__init__(node, switch=switch)

    def __call__(self, *operands):
        bypass = Bypass()
        self.options = [*self.options, bypass]
        return super().__call__(*operands)


class Bypass(Op):
    """Alternative Identity()"""

    def __init__(self) -> None:
        super().__init__(name="bypass")

    # def _verify_operands(self, *operands):
    #     assert len(operands) == 1

    def __call__(self, *operands) -> Any:
        op = super().__call__(*operands)
        # self._verify_operands(op.operands)
        return op

    def shape_fun(self):
        return self.operands[0].shape()

    def _forward_implementation(self, *operands):
        return operands[0].forward()
