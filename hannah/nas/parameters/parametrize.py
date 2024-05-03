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
import inspect
from copy import deepcopy
from inspect import Parameter as P
from typing import Optional, Sequence

from hannah.nas.core.expression import Expression
from hannah.nas.core.parametrized import is_parametrized
from hannah.nas.expressions.utils import extract_parameter_from_expression
from hannah.nas.parameters.parameters import Parameter


def _create_parametrize_wrapper(params, cls):
    parameter_list = list(params.values())
    old_init_fn = cls.__init__

    def init_fn(self, *args, **kwargs):
        if not hasattr(self, "_PARAMETERS"):
            self._PARAMETERS = {}

        # FIXME: Test what happens when the parent class is also @parametrized
        self._annotations = {}
        self._conditions = []
        self._condition_knobs = []

        num = 1
        tuple_idx = 0

        # FIXME: USE SIGNATURE NAME IF PARAM.NAME IS "" (empty)
        for arg in args:
            name = parameter_list[num].name
            if parameter_list[num].kind == P.VAR_POSITIONAL:
                name = name + f"_{tuple_idx}"
                tuple_idx += 1
            else:
                num += 1

            if is_parametrized(arg):
                self._PARAMETERS[name] = arg
                # arg.register()
            elif isinstance(arg, Expression):
                params = extract_parameter_from_expression(arg)
                idx = 0
                for p in params:
                    n = name + f"_{idx}"
                    self._PARAMETERS[n] = p
            elif isinstance(arg, Sequence):
                for idx, param in enumerate(arg):
                    if is_parametrized(param):
                        if not param.is_registered():
                            param.register()
                        n = f"{name}_{idx}"
                        self._PARAMETERS[n] = param
                    if isinstance(param, Expression):
                        params = extract_parameter_from_expression(param)
                        for idx2, p in enumerate(params):
                            if not p.is_registered():
                                p.register()
                            n = f"{name}_{idx}_{idx2}"
                            self._PARAMETERS[n] = p

        for name, arg in kwargs.items():
            if isinstance(arg, (list, tuple)):
                items = arg
            else:
                items = [arg]
            for i, item in enumerate(items):
                if is_parametrized(item):
                    if isinstance(item, Parameter):
                        if not item.is_registered():
                            self._PARAMETERS[item.name] = item
                            item.register()
                    else:
                        self._PARAMETERS[name] = item
                elif isinstance(item, Expression):
                    params = extract_parameter_from_expression(item)
                    for p in params:
                        if not p.is_registered():
                            self._PARAMETERS[p.name] = p
                            p.register()

        cls.sample = sample
        cls.set_current = set_current
        cls.instantiate = instantiate
        cls.check = check
        cls.set_params = set_params
        cls.parametrization = parametrization
        cls.get_parameters = get_parameters
        cls.set_param_scopes = set_param_scopes
        cls.cond = cond
        cls.add_param = add_param
        cls.get_constraints = get_constraints
        self._parametrized = True
        old_init_fn(self, *args, **kwargs)

    return init_fn


def parametrize(cls=None):
    def parametrize_function(cls):
        init_fn = cls.__init__
        init_sig = inspect.signature(init_fn)

        new_init_fn = _create_parametrize_wrapper(init_sig.parameters, cls)
        cls.__init__ = new_init_fn

        return cls

    if cls:
        return parametrize_function(cls)

    return parametrize_function


def sample(self, key=None):
    if key:
        for _key, param in self.parametrization(flatten=True).items():
            if key in _key:
                param.sample()
    else:
        for _key, param in self.parametrization(flatten=True).items():
            param.sample()


def set_current(self, value):
    self.set_params(**value)
    self.check(None)  # argument "value" not needed currently


def set_params(self, **kwargs):
    for key, value in kwargs.items():
        assert key in self._PARAMETERS, "{} has no parameter {}".format(self, key)

        if (
            not isinstance(value, dict)
            and key in self._annotations
            and not isinstance(value, self._annotations[key])
        ):
            raise TypeError(
                "Value must be of type {} but is {}".format(
                    self._annotations[key], type(value)
                )
            )
        if is_parametrized(value):
            self._PARAMETERS[key] = value
            setattr(self, key, value)  # TODO: Do we want this to work?
        else:
            self._PARAMETERS[key].set_current(value)


def check(self, value=None):
    for con in self._conditions:
        if not con.evaluate():
            raise Exception("Condition not satisfied: {}".format(con))


def cond(self, condition, allowed_params=None):
    self._conditions.append(condition)
    self._condition_knobs.append(allowed_params)


def get_constraints(self):
    constraints = []
    knobs = []
    queue = [self]
    visited = [self]

    while queue:
        n = queue.pop(-1)
        constraints.extend(n._conditions)
        knobs.extend(n._condition_knobs)
        for o in n.operands:
            # Cant use "in" because of EQ-Condition
            if o not in visited:
                queue.append(o)
                visited.append(o)
    return constraints, knobs


def instantiate(self):
    instance = deepcopy(self)
    instance._parametrized = False
    self.check(None)

    for key, param in instance._PARAMETERS.items():
        instantiated_value = param.instantiate()
        instance._PARAMETERS[key] = instantiated_value
        setattr(instance, key, instantiated_value)
    return instance


def add_param(self, id, param):
    assert id not in self._PARAMETERS, f"Parameter with the ID {id} already registered."
    param.id = id
    self._PARAMETERS[id] = param
    if hasattr(param, "register"):
        # if this is a Parameter, it must be registered
        param.register()
    return param


def get_parameters(
    self, scope: Optional[str] = None, include_empty=False, flatten=False
):
    params = {}
    visited = []
    queue = []
    queue.extend(self._PARAMETERS.values())

    while queue:
        current = queue.pop(-1)
        visited.append(current.id)
        if current.id is None:
            name = current.name
        else:
            name = current.id
        params[name] = current

        if hasattr(current, "_PARAMETERS"):
            for param in current._PARAMETERS.values():
                if param.id is None:
                    param.id = param.name
                if isinstance(param, Parameter) and param.id not in visited:
                    param.id = current.id + "." + param.name
                if param.id not in visited:
                    queue.append(param)

    params = hierarchical_parameter_dict(params, include_empty, flatten)
    return params


def parametrization(self, include_empty=False, flatten=True):
    return self.get_parameters(include_empty=include_empty, flatten=flatten)


def set_param_scopes(self):
    for name, param in self._PARAMETERS.items():
        if isinstance(param, Expression):
            param.id = self.id + "." + name
            param.set_scope(self.id, name)


def hierarchical_parameter_dict(parameter, include_empty=False, flatten=False):
    hierarchical_params = {}
    for key, param in parameter.items():
        if not include_empty and not isinstance(param, Parameter):
            continue
        if key is None:
            key_list = [param.name]
        else:
            key_list = key.split(".")
        if flatten:
            current_param_branch = {}
        else:
            current_param_branch = hierarchical_params

        for k in key_list:
            try:
                index = int(k)
                if index not in current_param_branch:
                    current_param_branch[index] = {}
                # current_param_branch = current_param_branch[index]
            except Exception:
                index = k
                if k not in current_param_branch:
                    current_param_branch[k] = {}

            if k == key_list[-1] and isinstance(param, Expression):
                if flatten:
                    if param.id is None:
                        hierarchical_params[param.name] = param
                    else:
                        hierarchical_params[param.id] = param
                else:
                    current_param_branch[index] = param
            else:
                current_param_branch = current_param_branch[index]
    return hierarchical_params


def set_parametrization(parameters, parametrization):
    for k, v in parametrization.items():
        if k in parameters:
            parametrization[k].set_current(parameters[k])
    return parametrization
