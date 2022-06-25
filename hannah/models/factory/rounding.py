import torch
from torch import Tensor


def round_downward(x: Tensor) -> Tensor:
    "Round to nearest upward"
    return torch.ceil(x - 0.5)


def round_upward(x: Tensor) -> Tensor:
    "Round to nearest downward"
    return torch.floor(x + 0.5)


def round_odd(x: Tensor) -> Tensor:
    "Round to nearest odd"
    return torch.round(x + 1.0) - 1.0


def round_even(x: Tensor) -> Tensor:
    "Round to nearest even"
    return torch.round(x)


def round_zero(x: Tensor) -> Tensor:
    "Round towards zero"
    return torch.where(x > 0, round_downward(x), round_upward(x))


def round_infinity(x: Tensor) -> Tensor:
    "Round toward infinity"
    return torch.where(x < 0, round_downward(x), round_upward(x))


# Truncating Rounding modes
def truncate_up(x: Tensor) -> Tensor:
    "Always round up to next integer"
    return torch.ceil(x)


def truncate_down(x: Tensor) -> Tensor:
    "Always round down to next integer"
    return torch.floor(x)


def truncate_infinity(x: Tensor) -> Tensor:
    "Always round to next integer in direction infinity"
    return torch.where(x < 0, torch.floor(x), torch.ceil(x))


def truncate_zero(x: Tensor) -> Tensor:
    "Always round to next integer in direction of Zero"
    return torch.where(x > 0, torch.floor(x), torch.ceil(x))


def round_stochastic(x: Tensor) -> Tensor:
    "Round stochastically"
    probs = x - torch.floor(x)
    return torch.floor(x) + torch.bernoulli(probs)


_MODE_MAP = {
    "DOWNWARD": round_downward,
    "UPWARD": round_upward,
    "ODD": round_odd,
    "EVEN": round_even,
    "ZERO": round_zero,
    "INFINITY": round_infinity,
    "STOCHASTIC": round_stochastic,
    "TRUNC_DOWN": truncate_down,
    "TRUNC_UP": truncate_up,
    "TRUNC_ZERO": truncate_zero,
    "TRUNC_INFINITY": truncate_infinity,
}


class RoundingMode:
    def __init__(self, mode: str):
        self.mode = mode.upper()
        assert mode in _MODE_MAP
        self.func = _MODE_MAP[self.mode]

    def __call__(self, x: Tensor) -> Tensor:
        return self.func(x)
