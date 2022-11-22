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
from dataclasses import dataclass
from typing import Any, List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

cs = ConfigStore.instance()


@dataclass
class StepLRConf:
    _target_: str = "torch.optim.lr_scheduler.StepLR"
    step_size: int = 10
    gamma: float = 0.1
    last_epoch: int = -1


cs.store(group="scheduler", name="step", node=StepLRConf())


@dataclass
class MultiStepLRConf:
    _target_: str = "torch.optim.lr_scheduler.MultiStepLR"
    milestones: List[int] = MISSING
    gamma: float = 0.1
    last_epoch: int = -1


cs.store(group="scheduler", name="multistep", node=MultiStepLRConf())


@dataclass
class ExponentialLRConf:
    _target_: str = "torch.optim.lr_scheduler.ExponentialLR"
    gamma: Any = 0.1
    last_epoch: Any = -1


cs.store(group="scheduler", name="exponential", node=ExponentialLRConf())


@dataclass
class CosineAnnealingLRConf:
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
    T_max: Any = MISSING
    eta_min: Any = 0
    last_epoch: Any = -1


cs.store(group="scheduler", name="cosine", node=CosineAnnealingLRConf())


@dataclass
class ReduceLROnPlateauConf:
    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    mode: str = "min"
    factor: Any = 0.1
    patience: Any = 10
    verbose: Any = False
    threshold: Any = 0.0001
    threshold_mode: str = "rel"
    cooldown: Any = 0
    min_lr: Any = 0
    eps: Any = 1e-08


cs.store(group="scheduler", name="plateau", node=ReduceLROnPlateauConf())


@dataclass
class CyclicLRConf:
    _target_: str = "torch.optim.lr_scheduler.CyclicLR"
    base_lr: Any = MISSING
    max_lr: Any = MISSING
    step_size_up: Any = 2000
    step_size_down: Any = None
    mode: str = "triangular"
    gamma: Any = 1.0
    scale_fn: Any = None
    scale_mode: str = "cycle"
    cycle_momentum: Any = True
    base_momentum: Any = 0.8
    max_momentum: Any = 0.9
    last_epoch: Any = -1


cs.store(group="scheduler", name="cyclic", node=CyclicLRConf())


@dataclass
class CosineAnnealingWarmRestartsConf:
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"
    T_0: Any = MISSING
    T_mult: Any = 1
    eta_min: Any = 0
    last_epoch: Any = -1


cs.store(group="scheduler", name="cosine_warm", node=CosineAnnealingWarmRestartsConf())


@dataclass
class OneCycleLRConf:
    "Config for one cycle lr total steps are configured from module"
    _target_: str = "torch.optim.lr_scheduler.OneCycleLR"
    max_lr: float = "${optimizer.lr}"
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    cycle_momentum: Any = True
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 10000.0
    last_epoch: int = -1


cs.store(group="scheduler", name="1cycle", node=OneCycleLRConf())
