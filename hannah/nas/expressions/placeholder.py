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
from copy import deepcopy
from typing import Optional

from ..core.expression import Expression


class Placeholder(Expression):
    def __init__(self, id: Optional[str] = None):
        self.id = id
        self._conditions = []

    def get_children(self):
        return []

    def evaluate(self):
        raise NotImplementedError()

    def new(self):
        return deepcopy(self)

    def format(self, indent=2, length=80) -> str:
        return self.__class__.__name__ + "()"


class UndefinedInt(Placeholder):
    def __init__(self, id: Optional[str] = None) -> None:
        super().__init__(id)


# TODO:
class UndefinedFloat(Placeholder):
    def __init__(self, id: Optional[str] = None) -> None:
        super().__init__(id)


class DefaultInt(Placeholder):
    def __init__(self, value: int, id: Optional[str] = None) -> None:
        super().__init__(id)
        self.value = value

    def evaluate(self):
        return self.value

    def format(self, indent=2, length=80) -> str:
        return self.__class__.__name__ + "({})".format(self.value)

    def __repr__(self) -> str:
        return str(self.value)


class DefaultFloat(Placeholder):
    def __init__(self, value: float, id: Optional[str] = None) -> None:
        super().__init__(id)
        self.value = value


class DefaultBool(Placeholder):
    def __init__(self, value: bool, id: Optional[str] = None) -> None:
        super().__init__(id)
        self.value = value


class IntRange(Placeholder):
    def __init__(self, min: int, max: int, id: Optional[str] = None) -> None:
        super().__init__(id)
        self.min = min
        self.max = max
        # TODO: self.value = ??


class FloatRange(Placeholder):
    def __init__(self, lower: int, upper: int, id: Optional[str] = None):
        super().__init__(id)
        self.lower = lower
        self.upper = upper


class Categorical(Placeholder):
    def __init__(self, id: Optional[str] = None):
        super().__init__(id)
