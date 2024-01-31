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
import logging

from z3 import And, AtLeast, AtMost, Bool, If, Implies, Int, Or, Real, Solver

from hannah.nas.expressions.arithmetic import Add, Floor, Floordiv, Mul, Sub, Truediv
from hannah.nas.expressions.conditions import (
    GECondition,
    GTCondition,
    LECondition,
    LTCondition,
)
from hannah.nas.expressions.logic import And as expr_and
from hannah.nas.expressions.logic import If as expr_if
from hannah.nas.expressions.op import BinaryOp, UnaryOp
from hannah.nas.expressions.placeholder import (
    Categorical,
    DefaultInt,
    IntRange,
    Placeholder,
)
from hannah.nas.parameters.parameters import (
    CategoricalParameter,
    IntScalarParameter,
    Parameter,
)

logger = logging.getLogger(__name__)


class ConstraintModel:
    def __init__(self, method="naive", fix_strategy="none", timeout=300) -> None:
        self.method = method
        self.fix_strategy = fix_strategy  # FIXME: Implement
        self.timeout = timeout  # currently not in use
        self.solver = []
        self.vars = {}

        self.input_dict = {}
        self.output_dict = {}
        self.enter_dict = {}

    def build_model(self, conditions, fixed_vars=[]):
        for con in conditions:
            sol = Solver()
            self.vars[sol] = {}
            sol.add(self.build_constraint_from_expression(sol, con, fixed_vars))
            self.solver.append(sol)

    def extract_parameter(self, solver, expr, fixed_vars):
        if isinstance(expr, (IntScalarParameter, IntRange)):
            return self.extract_int_range(solver, expr, fixed_vars)
        elif isinstance(expr, (CategoricalParameter, Categorical)):
            return self.extract_categorical(solver, expr, fixed_vars)
        elif isinstance(expr, DefaultInt):
            return self.extract_defaultint(solver, expr)
        elif isinstance(expr, int):
            var = Int(expr.id)
            solver.add(var == expr)
            return var

    def fix_var(self, solver, param):
        try:
            self.extract_parameter(solver, param, [param.id])
        except Exception as e:
            logger.critical(str(e))

    def get_tracker_var(self, solver, expr, key=None):
        tracker_var = Bool(f"tracker_{expr.id}")
        if key and key not in expr.id:
            return None
        try:
            var = Int(expr.id)
            con = var == int(expr.current_value)
            solver.add(Implies(tracker_var, con))
        except Exception:
            pass
        return tracker_var

    def get_all_tracker_vars(self, solver, module, key=None, parameters=None):
        tracker_vars = []
        if not parameters:
            parameters = module.parametrization(flatten=True)
        for n, p in parameters.items():
            v = self.get_tracker_var(solver, p, key)
            if v is not None:
                tracker_vars.append(v)
        return tracker_vars

    def linear_search(self, solver, trackers):
        for k in reversed(range(len(trackers))):
            solver.push()
            solver.add(AtLeast(*trackers, k))
            res = solver.check()
            if res.r != -1:
                print(f"Can satisfy {k} soft constraints.")
                return
            else:
                solver.pop()
                # print("Not sat")
        raise Exception("No satisfiable configuration possible")

    def naive_search(self, solver, module, key=None, parameters=None):
        ct = 0
        if not parameters:
            parameters = module.parametrization(flatten=True)
        for n, p in parameters.items():
            if key and key not in n:
                continue
            if n not in self.vars[solver]:
                continue
            try:
                var = Int(n)
                if hasattr(p, "current_value"):
                    val = int(p.current_value)
                else:
                    val = int(p)
                con = var == val
                solver.push()
                ct += 1
                solver.add(con)
            except Exception:
                pass
        for i in range(ct):
            res = solver.check()
            if res.r == 1:
                solver.pop()
                return
            else:
                solver.pop()

        raise Exception("No satisfiable configuration possible")

    def solve(self, module, parameters=None, key=None, fix_vars=[]):
        self.soft_constrain_current_parametrization(module, parameters, key, fix_vars)

    def soft_constrain_current_parametrization(self, module, parameters=None, key=None, fix_vars=[]):
        self.solver = []
        self.build_model(module._conditions)
        for solver in self.solver:
            for v in fix_vars:
                self.fix_var(solver, v)
            if self.method == "linear":
                trackers = self.get_all_tracker_vars(solver, module, key, parameters)
                self.linear_search(solver, trackers)
            elif self.method == "naive":
                self.naive_search(solver, module, key, parameters)

    def get_constrained_params(self, params: dict):
        for solver in self.solver:
            solver.check()
            mod = solver.model()
            for name, p in params.items():
                if name in self.vars[solver]:
                    try:
                        params[name] = mod[self.vars[solver][name]].as_long()
                    except Exception as e:
                        logger.error(str(e))
        return params

    def insert_model_values_to_module(self, module):
        for solver in self.solver:
            solver.check()
            mod = solver.model()
            for name, p in module.parametrization(flatten=True).items():
                if name in self.vars[solver]:
                    try:
                        value = mod[self.vars[solver][name]].as_long()
                        p.current_value = value
                    except Exception as e:
                        print(str(e))
        return module

    def extract_int_range(self, solver, expr, fixed_vars):
        if expr.id:
            var = Int(expr.id)
        else:
            var = Int(f"IntRange({expr.min}, {expr.max})")
            # TODO: unique scope ids for DFG parameters
        self.vars[solver][str(var)] = var
        if expr.id in fixed_vars:
            solver.add(var == expr.current_value)
            return var
        solver.add(var >= expr.min)
        solver.add(var <= expr.max)
        if hasattr(expr, "step_size") and expr.step_size != 1:
            solver.add((var - expr.min) % expr.step_size == 0)

        return var

    def extract_categorical(self, solver, expr, fixed_vars):
        var = Int(expr.id)
        self.vars[solver][expr.id] = var
        if expr.id in fixed_vars:
            solver.add(var == int(expr.current_value))
            return var
        cons = []
        for val in expr.choices:
            cons.append(var == val)
        solver.add(Or(cons))
        return var

    def extract_defaultint(self, solver, expr):
        if expr.id:
            var = Int(expr.id)
        else:
            var = Int(f"DefaultInt({expr.value})")
        self.vars[solver][str(var)] = var
        solver.add(var == expr.value)
        return var

    def build_constraint_from_expression(self, solver, expr, fixed_vars=[]):
        if isinstance(expr, Parameter):
            var = self.extract_parameter(solver, expr, fixed_vars)
            self.vars[solver][str(var)] = var
            return var
        elif isinstance(expr, Placeholder):
            var = self.extract_parameter(solver, expr)
            return var
        elif isinstance(expr, int):
            var = Int(f"Literal({expr})")
            solver.add(var == expr)
            return var
        elif isinstance(expr, float):
            if expr.is_integer():
                var = Int(f"Literal({expr})")
            else:
                var = Real(f"Literal({expr})")
            solver.add(var == expr)
            return var
        elif isinstance(expr, Floor):
            con = self.build_constraint_from_expression(
                solver, expr.operand, fixed_vars
            )
            return con
        elif isinstance(expr, expr_if):
            operand = self.build_constraint_from_expression(solver, expr.operand)
            a = self.build_constraint_from_expression(solver, expr.a)
            b = self.build_constraint_from_expression(solver, expr.b)
            con = If(operand, a, b)
            return con
        elif isinstance(expr, BinaryOp):
            lhs = self.build_constraint_from_expression(solver, expr.lhs, fixed_vars)
            rhs = self.build_constraint_from_expression(solver, expr.rhs, fixed_vars)
            if isinstance(expr, Add):
                con = lhs + rhs
            elif isinstance(expr, Truediv):
                con = lhs / rhs
            elif isinstance(expr, Floordiv):
                con = lhs / rhs
            elif isinstance(expr, Mul):
                con = lhs * rhs
            elif isinstance(expr, Sub):
                con = lhs - rhs
            elif isinstance(expr, LECondition):
                con = lhs <= rhs
            elif isinstance(expr, LTCondition):
                con = lhs < rhs
            elif isinstance(expr, GECondition):
                con = lhs >= rhs
            elif isinstance(expr, GTCondition):
                con = lhs > rhs
            elif isinstance(expr, expr_and):
                con = And(lhs, rhs)
            return con
        else:
            raise Exception(
                f"The expression -> constraint transformation is not defined for: {expr} of type {type(expr)}."
            )


def check_for_id(a, b):
    return hasattr(a, "id") and hasattr(b, "id") and a.id and b.id and a.id == b.id


if __name__ == "__main__":
    cm = ConstraintModel()
    print()
