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
from math import floor, ceil

from .op import BinaryOp, UnaryOp


class Add(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "+"

    def concrete_impl(self, lhs, rhs):
        return lhs + rhs


class Sub(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "-"

    def concrete_impl(self, lhs, rhs):
        return lhs - rhs


class Mul(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "*"

    def concrete_impl(self, lhs, rhs):
        return lhs * rhs


class Truediv(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "/"

    def concrete_impl(self, lhs, rhs):
        return lhs / rhs


class Floordiv(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "//"

    def concrete_impl(self, lhs, rhs):
        return lhs // rhs


class Mod(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "mod"

    def concrete_impl(self, lhs, rhs):
        return lhs % rhs


class Floor(UnaryOp):
    def __init__(self, operand) -> None:
        super().__init__(operand)

    def concrete_impl(self, operand):
        return floor(operand)


class Ceil(UnaryOp):
    def __init__(self, operand) -> None:
        super().__init__(operand)

    def concrete_impl(self, operand):
        return ceil(operand)
