import pytest
import torch

from hannah.models.factory.rounding import RoundingMode


@pytest.mark.parametrize(
    "mode,expected",
    [
        #                  [-1.9,-1.5,-1.0,-0.5,0.0,0.5,1.5,1.9,0.49,1.51]
        ("DOWNWARD", [-2.0, -2.0, -1.0, -1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0]),
        ("UPWARD", [-2.0, -1.0, -1.0, -0.0, 0.0, 1.0, 2.0, 2.0, 0.0, 2.0]),
        ("ODD", [-2.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 2.0, 0.0, 2.0]),
        ("EVEN", [-2.0, -2.0, -1.0, -0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0]),
        ("ZERO", [-2.0, -1.0, -1.0, -0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0]),
        ("INFINITY", [-2.0, -2.0, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 0.0, 2.0]),
        ("TRUNC_DOWN", [-2.0, -2.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
        ("TRUNC_UP", [-1.0, -1.0, -1.0, -0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 2.0]),
        ("TRUNC_ZERO", [-1.0, -1.0, -1.0, -0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
        ("TRUNC_INFINITY", [-2.0, -2.0, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 1.0, 2.0]),
    ],
)
def test_rounding(mode, expected):
    data = torch.tensor([-1.9, -1.5, -1.0, -0.5, 0.0, 0.5, 1.5, 1.9, 0.49, 1.51])
    round = RoundingMode(mode)
    assert torch.allclose(round(data), torch.tensor(expected))


def test_stochastic_rounding():
    round = RoundingMode("STOCHASTIC")
