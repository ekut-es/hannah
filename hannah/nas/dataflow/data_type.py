from abc import ABC, abstractmethod
from typing import Tuple, Union
from numbers import Number


class DataType(ABC):
    @abstractmethod
    def bits(self) -> int:
        ...

    @abstractmethod
    def range(self) -> Tuple[Number, Number]:
        ...


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
            min_val = -2 ** (self.bits - 1)
            max_val = 2 ** (self.bits - 1) - 1
        else:
            min_val = 0
            max_val = 2 ** (self.bits) - 1

        return (min_val, max_val)


class FloatType(DataType):
    def __init__(self, signed=True, significand_bits=23, exponent_bits=8):
        self.signed = signed
        self.significand_bits = significand_bits
        self.exponent_bits = exponent_bits

    def bits(self) -> int:
        bits = self.significand + self.exponent_bits
        if self.signed:
            bits += 1

        return bits

    def signed(self) -> int:
        return self.signed

    def range(self) -> float:
        # FIXME: calculate correct range
        reserved_bits = 2
        exponent_bias = (2 ** self.exponent_bits - reserved_bits) / 2
        max_val = (2 - 2 ** (-self.significand_bits)) * 2 ** (
            self.exponent_bits - exponent_bias
        )
        min_val = -1 * max_val
        return (min_val, max_val)


if __name__ == "__main__":
    fl = FloatType()
    rn = fl.range()
    print(rn)
