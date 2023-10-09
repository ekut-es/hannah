#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
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
from hannah.nas.core.expression import Expression
from .op import BinaryOp, UnaryOp


class And(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "and"

    def concrete_impl(self, lhs, rhs):
        # Fixme this does not follow python semantacics exactly
        return lhs and rhs


class Or(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "or"

    def concrete_impl(self, lhs, rhs):
        # Fixme this does not follow python semantacics exactly
        return lhs or rhs


# FIXME: Just want to try whether this works in z3
class If(Expression):
    def __init__(self, operand, a, b) -> None:
        super().__init__()
        self.operand = operand
        self.a = a
        self.b = b

    def get_children(self):
        return [self.operand, self.a, self.b]

    def evaluate(self):
        condition = self.operand
        if hasattr(self.operand, 'evaluate'):
            condition = self.operand.evaluate()
        result = self.a if condition else self.b
        if hasattr(result, 'evaluate'):
            result = result.evaluate()
        return result

    def format(self, indent=2, length=80):
        return f"If({self.operand}, {self.a}, {self.b})"


    def __repr__(self):
        return "If(" + repr(self.operand) + ", " + repr(self.a) + ", " + repr(self.b) + ")"


