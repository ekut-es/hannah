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
from typing import TYPE_CHECKING, Dict, Protocol, TypeVar, Union, runtime_checkable

if TYPE_CHECKING:
    from ..parameters.parameters import Parameter

T = TypeVar("T")

ParameterTree = Union["Parameter", Dict[str, "ParameterTree"]]


@runtime_checkable
class Parametrized(Protocol[T]):
    def sample(self) -> None:
        ...

    def instantiate(self) -> T:
        ...

    def set_current(self, value) -> None:
        ...

    def check(self, value) -> bool:
        ...

    def parametrization(self) -> ParameterTree:
        ...


def is_parametrized(obj):
    return isinstance(obj, Parametrized)
