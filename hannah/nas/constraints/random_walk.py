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
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from hannah.nas.functional_operators.lazy import lazy
from hannah.nas.functional_operators.op import ChoiceOp
from hannah.nas.parameters.parameters import Parameter
from hannah.nas.parameters.parametrize import set_parametrization
from hannah.nas.search.utils import np_to_primitive

logger = logging.getLogger(__name__)


@dataclass
class Param:
    name: str
    value: Any


def hierarchical_parameter_dict(parameter, include_empty=False, flatten=False):
    hierarchical_params = {}
    for key, param in parameter.items():
        key_list = key.split(".")
        key_list = key_list[:2] + [".".join(key_list[2:])]
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

            if k == key_list[-1]:
                current_param_branch[index] = Param(name=key, value=param)
            else:
                current_param_branch = current_param_branch[index]
    return hierarchical_params


CHOICES = {
    0: "conv",
    1: "expand_reduce",
    2: "reduce_expand",
    3: "pooling",
    4: "sandglass",
}


def get_active_parameter(net):
    active_param_ids = []
    queue = [net]
    visited = [net.id]

    def extract_parameters(node):
        ids = []
        for k, p in node._PARAMETERS.items():
            if isinstance(p, Parameter):
                ids.append(p.id)
        return ids

    while queue:
        current = queue.pop()
        if isinstance(current, ChoiceOp):
            # handle choices
            active_param_ids.append(current.switch.id)
            chosen_path = current.options[lazy(current.switch)]
            if chosen_path.id not in visited:
                queue.append(chosen_path)
                visited.append(chosen_path.id)
        else:
            # handle all other operators & tensors
            active_param_ids.extend(extract_parameters(current))
            for operand in current.operands:
                if operand.id not in visited:
                    queue.append(operand)
                    visited.append(operand.id)

    return active_param_ids


class RandomWalkConstraintSolver:
    def __init__(self, max_iterations=5000) -> None:
        self.max_iterations = max_iterations
        self.constraints = None
        self.solution = None

    def build_model(self, conditions, fixed_vars=[]):
        self.constraints = conditions

    # def solve(self, module, parameters, fix_vars=[]):
    #     mod = deepcopy(module)
    #     self.solution = deepcopy(parameters)
    #     params = deepcopy(parameters)

    #     solved_conditions = []

    #     for i, con in enumerate(mod._conditions):
    #         param_keys = list(params.keys())
    #         if mod._condition_knobs[i] is not None:
    #             param_keys = [p.id for p in mod._condition_knobs]
    #         ct = 0
    #         while ct < self.max_iterations:
    #             key_to_change = random.choice(param_keys)
    #             old_val = mod.parametrization(flatten=True)[key_to_change].current_value
    #             new_val = mod.parametrization(flatten=True)[key_to_change].sample()
    #             try:
    #                 # first, assure that the proposed solution for the new constraint does not violate already solved constraints
    #                 try:
    #                     for c in solved_conditions:
    #                         c.evaluate()
    #                         print("Solution violated already satisfied constraint")
    #                 except Exception:
    #                     mod.parametrization(flatten=True)[key_to_change].set_current(old_val)
    #                 con.evaluate()
    #                 params[key_to_change] = new_val
    #                 self.solution.update(params)
    #                 solved_conditions.append(con)
    #                 print(f"Solved constraint {i} with {ct} iterations.")
    #                 break
    #             except Exception:
    #                 params[key_to_change] = new_val
    #                 ct += 1
    #         print(f"Failed to solve constraint {i}.")

    def right_direction(self, current, new, direction):
        if direction == ">":
            if new > current:
                return True
            else:
                return False
        elif direction == "<":
            if new < current:
                return True
            else:
                return False

    def solve(self, module, parameters, fix_vars=[]):
        print("Start constraint solving")
        mod = deepcopy(module)  # FIXME copying is inefficient
        # mod = module
        self.solution = deepcopy(parameters)
        params = deepcopy(parameters)
        set_parametrization(parameters, mod.parametrization(flatten=True))

        solved_conditions = []
        constraints, knobs = mod.get_constraints()
        constraints = list(reversed(constraints))
        knobs = list(reversed(knobs))

        for i, con in enumerate(constraints):
            all_param_keys = list(params.keys())
            if knobs[i] is not None:
                all_param_keys = [p.id for p in knobs[i]]

            direction = con.symbol

            ct = 0
            while ct < self.max_iterations:
                # active_params = get_active_parameter(params)
                active_params = get_active_parameter(mod)

                param_keys = [p for p in all_param_keys if p in active_params]
                current = con.lhs.evaluate()
                if con.evaluate():
                    self.solution.update(params)
                    solved_conditions.append(con)
                    logger.info(f"Solved constraint {i} with {ct} iterations.")
                    break
                else:
                    new_target = lazy(con.lhs)
                    key_to_change = random.choice(param_keys)
                    old_val = mod.parametrization(flatten=True)[
                        key_to_change
                    ].current_value
                    new_val = mod.parametrization(flatten=True)[key_to_change].sample()

                    j = 0
                    while not self.right_direction(current, new_target, direction):
                        mod.parametrization(flatten=True)[key_to_change].set_current(
                            old_val
                        )

                        key_to_change = random.choice(param_keys)
                        old_val = mod.parametrization(flatten=True)[
                            key_to_change
                        ].current_value
                        new_val = mod.parametrization(flatten=True)[
                            key_to_change
                        ].sample()
                        new_target = lazy(con.lhs)
                        # param_keys.remove(key_to_change)
                        j += 1
                        if j > 1000:
                            raise Exception("Timeout: Failed to find improvement.")
                        if len(param_keys) == 0:
                            break  # FIXME: break or fail?

                    # print(f"Param: {key_to_change}: {old_val} -> {new_val}")
                    valid = True
                    new_target = con.lhs.evaluate()
                    if self.right_direction(current, new_target, direction):
                        for c in solved_conditions:
                            if not c.evaluate():
                                print("Solution violated already satisfied constraint")
                                # reverse modification to satisfy already solved constraints again
                                param_keys.remove(key_to_change)
                                mod.parametrization(flatten=True)[
                                    key_to_change
                                ].set_current(old_val)
                                valid = False
                    else:
                        print("No improvement")
                        mod.parametrization(flatten=True)[key_to_change].set_current(
                            old_val
                        )
                        valid = False
                    if valid:
                        # update proposed solution for this constraint
                        params[key_to_change] = new_val
                    print(f"Constraint lhs: {lazy(con.lhs)} - rhs: {lazy(con.rhs)}")
                    ct += 1
                    if ct == self.max_iterations - 1:
                        print(f"Failed to solve constraint {i}.")
                        raise Exception(f"Failed to solve constraint {i}.")

    def get_constrained_params(self, params: dict):
        return np_to_primitive(self.solution)
