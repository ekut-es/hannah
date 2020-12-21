from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, List

from hydra.core.config_store import ConfigStore


@dataclass
class ChoiceList:
    choices: List[Any]
    min: int = 4
    max: int = 10


@dataclass
class ScalarConfigSpec:
    """Representation of all the options to define
    a scalar.
    """

    # lower bound if any
    lower: Optional[float] = None

    # upper bound if any
    upper: Optional[float] = None

    # initial value
    # default to the middle point if completely bounded
    init: Optional[float] = None

    # step size for an update
    # defaults to 1 if unbounded
    # or 1/6 of the range if completely bounded
    step: Optional[float] = None

    # cast to integer
    integer: bool = False

    # logarithmically distributed
    log: bool = False


@dataclass
class OptimConf:
    optimizer: str = "aging_evolution"
    population_size: int = 100
    sample_size: int = 25
    budget: int = 2000
    eps: float = 0.1
    num_workers: int = 10
    seed: Optional[int] = None


@dataclass
class UNASSweeperConf:
    _target_: str = "hydra_plugins.hydra_unas_sweeper.unas_sweeper.UNASSweeper"
    optim: OptimConf = OptimConf()
    parametrization: Dict[str, Any] = field(default_factory=dict)


ConfigStore.instance().store(
    name="unas", node=UNASSweeperConf, group="hydra/sweeper", provider="unas"
)
