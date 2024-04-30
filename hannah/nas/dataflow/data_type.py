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
from abc import ABC, abstractmethod
from numbers import Number
from typing import Tuple, Union


class DataType(ABC):
    @abstractmethod
    def bits(self) -> int:
        ...

    @abstractmethod
    def range(self) -> Tuple[Number, Number]:
        ...

    def as_numpy(self) -> str:
        return ""


class IntType(DataType):
    def __init__(self, signed: bool = True, bits: int = 8):
        self.signed = signed
        self.bits = bits

    def bits(self) -> int:
        return self.bits

    def signed(self) -> bool:
        return self.signed

    def range(self) -> Tuple[int, int]:
        if self.signed:
            min_val = -(2 ** (self.bits - 1))
            max_val = 2 ** (self.bits - 1) - 1
        else:
            min_val = 0
            max_val = 2 ** (self.bits) - 1

        return (min_val, max_val)

    def as_numpy(self) -> str:
        if self.signed:
            return f"np.int{self.bits}"
        else:
            return f"np.uint{self.bits}"


class FloatType(DataType):
    def __init__(self, signed=True, significand_bits=23, exponent_bits=8):
        self.signed = signed
        self.significand_bits = significand_bits
        self.exponent_bits = exponent_bits

    def bits(self) -> int:
        bits = self.significand_bits + self.exponent_bits
        if self.signed:
            bits += 1

        return bits

    def signed(self) -> int:
        return self.signed

    def range(self) -> float:
        # FIXME: calculate correct range
        reserved_bits = 2
        exponent_bias = (2**self.exponent_bits - reserved_bits) / 2
        max_val = (2 - 2 ** (-self.significand_bits)) * 2 ** (
            self.exponent_bits - exponent_bias
        )
        min_val = -1 * max_val
        return (min_val, max_val)

    def as_numpy(self) -> str:
        return f"float{self.bits()}"


if __name__ == "__main__":
    fl = FloatType()
    rn = fl.range()
    print(rn)
