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
    def bits(self) -> int: ...

    @abstractmethod
    def range(self) -> Tuple[Number, Number]: ...

    def as_numpy(self) -> str:
        return ""


class IntType(DataType):
    """Describe the properties of an integer datatype.add()

        Args:
            signed (bool, optional): Whether the integer is signed or not. Defaults to True.
            bits (int, optional): The number of bits used to represent the integer. Defaults to 8.
            reduce_range (bool, optional): Whether to reduce the range of the integer to make the dataranges symmetric around zero.  Only applies to signed datatypes. Defaults to Fa
    lse.
    """

    def __init__(self, signed: bool = True, bits: int = 8, reduce_range=False):
        self._signed = signed
        self._bits = bits
        self._reduce_range = reduce_range

    def bits(self) -> int:
        return self._bits

    def signed(self) -> bool:
        return self._signed

    def range(self) -> Tuple[int, int]:
        if self._signed:
            min_val = -(2 ** (self._bits - 1))

            if self._reduce_range:
                min_val += 1
            max_val = 2 ** (self._bits - 1) - 1
        else:
            min_val = 0
            max_val = 2 ** (self._bits) - 1

        return (min_val, max_val)

    def as_numpy(self) -> str:
        if self.signed:
            return f"np.int{self._bits}"
        else:
            return f"np.uint{self._bits}"

    def __str__(self):
        return f"{'u' if not self._signed else 'i'}{self._bits}"


class FloatType(DataType):
    def __init__(self, signed=True, significand_bits=23, exponent_bits=8):
        self._signed = signed
        self._significand_bits = significand_bits
        self._exponent_bits = exponent_bits

    def bits(self) -> int:
        bits = self._significand_bits + self._exponent_bits
        if self._signed:
            bits += 1

        return bits

    def signed(self) -> int:
        return self._signed

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

    def __str__(self):
        return f"f{self.bits()}"


if __name__ == "__main__":
    fl = FloatType()
    rn = fl.range()
    print(rn)
