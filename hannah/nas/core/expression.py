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
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Expression(ABC, Generic[T]):
    @abstractmethod
    def evaluate(self):
        ...

    @abstractmethod
    def format(self, indent=2, length=80) -> str:
        ...

    def __str__(self) -> str:
        return self.format()

    def __lt__(self, other):
        from ..expressions.conditions import LTCondition

        return LTCondition(self, other)

    def __le__(self, other):
        from ..expressions.conditions import LECondition

        return LECondition(self, other)

    def __gt__(self, other):
        from ..expressions.conditions import GTCondition

        return GTCondition(self, other)

    def __ge__(self, other):
        from ..expressions.conditions import GECondition

        return GECondition(self, other)

    def __eq__(self, other):
        from ..expressions.conditions import EQCondition

        return EQCondition(self, other)

    def __ne__(self, other):
        from ..expressions.conditions import NECondition

        return NECondition(self, other)

    def __add__(self, other):
        from ..expressions.arithmetic import Add

        return Add(self, other)

    def __radd__(self, other):
        from ..expressions.arithmetic import Add

        return Add(other, self)

    def __sub__(self, other):
        from ..expressions.arithmetic import Sub

        return Sub(self, other)

    def __mul__(self, other):
        from ..expressions.arithmetic import Mul

        return Mul(self, other)

    def __rmul__(self, other):
        from ..expressions.arithmetic import Mul

        return Mul(other, self)

    def __truediv__(self, other):
        from ..expressions.arithmetic import Truediv

        return Truediv(self, other)

    def __floordiv__(self, other):
        from ..expressions.arithmetic import Floordiv

        return Floordiv(self, other)

    def __mod__(self, other):
        from ..expressions.arithmetic import Mod

        return Mod(self, other)

    def __divmod__(self, other):
        raise NotImplementedError()

    def __pow__(self, other):
        raise NotImplementedError()

    def __lshift__(self, other):
        raise NotImplementedError()

    def __rshift__(self, other):
        raise NotImplementedError()

    def __and__(self, other):
        from ..expressions.logic import And

        return And(self, other)

    def __xor__(self, other):
        raise NotImplementedError()

    def __or__(self, other):
        from ..expressions.logic import Or

        return Or(self, other)

    def set_scope(self, scope, name=""):
        self.id = f"{scope}.{name}"
