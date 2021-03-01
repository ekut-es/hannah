from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, List

from hydra.core.config_store import ConfigStore

@dataclass
class Choice:
    choices: List[Any]

@dataclass
class Partition:
    choices: List[Any]
    partitions: int 

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
    

@dataclass
class OptimConf:
    optimizer: str = "aging_evolution"
    surrogate: Optional[XGBoostSurrogate] = None
    population_size: int = 100
    sample_size: int = 25
    budget: int = 2000
    eps: float = 0.1
    num_workers: int = 10
    seed: Optional[int] = None
    bounds: Optional[Dict[str, float]] = None


@dataclass
class UNASSweeperConf:
    _target_: str = "hydra_plugins.hydra_unas_sweeper.unas_sweeper.UNASSweeper"
    optim: OptimConf = OptimConf()
    parametrization: Dict[str, Any] = field(default_factory=dict)


ConfigStore.instance().store(
    name="unas", node=UNASSweeperConf, group="hydra/sweeper", provider="unas"
)
