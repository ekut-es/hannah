from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class Choice:
    choices: List[Any]


@dataclass
class Partition:
    choices: List[Any]
    partitions: int
    ordered: bool = True


@dataclass
class Subset:
    choices: List[Any]
    size: int


@dataclass
class ChoiceList:
    choices: List[Any]
    min: int
    max: int


@dataclass
class Scalar:
    """Representation of all the options to define
    a scalar.
    """

    # lower bound if any
    lower: Optional[float] = None

    # upper bound if any
    upper: Optional[float] = None

    # step size for an update
    # defaults to 1 if unbounded
    # or 1/6 of the range if completely bounded
    step: Optional[float] = None

    # cast to integer
    integer: bool = False

    # logarithmically distributed
    log: bool = False


@dataclass
class XGBoostSurrogate:
    candidates: int = 25
    booster: str = "gboost"
    objective: str = "reg:squarederror"
    eta: float = 1.0
    gamma: float = 1.0
    min_child_weight: float = 1.0
    max_depth: int = 25
