from copy import deepcopy
import random
from typing import Any
import numpy as np
from hannah.nas.parameters.parametrize import set_parametrization
from hannah.nas.search.utils import np_to_primitive


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

    def solve(self, module, parameters, fix_vars=[]):
        mod = deepcopy(module)
        self.solution = deepcopy(parameters)
        params = deepcopy(parameters)
        set_parametrization(parameters, mod.parametrization(flatten=True))

        solved_conditions = []
        constraints, knobs = mod.get_constraints()
        constraints = list(reversed(constraints))
        knobs = list(reversed(knobs))

        for i, con in enumerate(constraints):
            param_keys = list(params.keys())
            if knobs[i] is not None:
                param_keys = [p.id for p in knobs[i]]

            ct = 0
            while ct < self.max_iterations:
                if con.evaluate():
                    self.solution.update(params)
                    solved_conditions.append(con)
                    print(f"Solved constraint {i} with {ct} iterations.")
                    break
                else:
                    key_to_change = random.choice(param_keys)
                    old_val = mod.parametrization(flatten=True)[key_to_change].current_value
                    new_val = mod.parametrization(flatten=True)[key_to_change].sample()

                    valid = True
                    for c in solved_conditions:
                        if not c.evaluate():
                            print("Solution violated already satisfied constraint")
                            # reverse modification to satisfy already solved constraints again
                            mod.parametrization(flatten=True)[key_to_change].set_current(old_val)
                            valid = False
                    if valid:
                        # update proposed solution for this constraint
                        params[key_to_change] = new_val
                    ct += 1
                    if ct == self.max_iterations-1:
                        print(f"Failed to solve constraint {i}.")
                        raise Exception(f"Failed to solve constraint {i}.")

    def get_constrained_params(self, params: dict):
        return np_to_primitive(self.solution)
