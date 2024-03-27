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
from abc import ABC, abstractmethod, abstractproperty
from numbers import Number
from typing import Tuple, Union


class DataType(ABC):
    @abstractproperty
    def bits(self) -> int:
        ...

    @abstractmethod
    def range(self) -> Tuple[Number, Number]:
        ...


class IntType(DataType):
    """Describe the properties of an integer datatype.add()

    Args:
        signed (bool, optional): Whether the integer is signed or not. Defaults to True.
        bits (int, optional): The number of bits used to represent the integer. Defaults to 8.
        reduce_range (bool, optional): Whether to reduce the range of the integer to make the dataranges symmetric around zero.  Only applies to signed datatypes. Defaults to False.
    """

    def __init__(self, signed: bool = True, bits: int = 8, reduce_range: bool = False):
        self.signed = signed
        self._bits = bits
        self.reduce_range = reduce_range

    def signed(self) -> bool:
        return self.signed

    def range(self) -> Tuple[int, int]:
        if self.signed:
            if self.reduce_range:
                min_val = -(2 ** (self._bits - 1)) + 1
            else:
                min_val = -(2 ** (self._bits - 1))
            max_val = 2 ** (self._bits - 1) - 1
        else:
            min_val = 0
            max_val = 2 ** (self._bits) - 1

        return (min_val, max_val)

    def __str__(self):
        return f"u{self._bits}" if not self.signed else f"i{self._bits}"

    @property
    def bits(self) -> int:
        return self._bits


class FloatType(DataType):
    def __init__(self, signed=True, significand_bits=23, exponent_bits=8):
        self.signed = signed
        self.significand_bits = significand_bits
        self.exponent_bits = exponent_bits

    @property
    def bits(self) -> int:
        bits = self.significand_bits + self.exponent_bits
        if self.signed:
            bits += 1

        return bits

    def range(self) -> float:
        # FIXME: calculate correct range
        reserved_bits = 2
        exponent_bias = (2**self.exponent_bits - reserved_bits) / 2
        max_val = (2 - 2 ** (-self.significand_bits)) * 2 ** (
            self.exponent_bits - exponent_bias
        )
        min_val = -1 * max_val
        return (min_val, max_val)

    def __str__(self):
        return f"f{self._bits}"


if __name__ == "__main__":
    fl = FloatType()
    rn = fl.range()
    print(rn)
