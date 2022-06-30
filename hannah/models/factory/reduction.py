import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.container import Sequential


class ReductionBlockAdd(nn.Module):
    """Reduction block that sums over its inputs"""

    def __init__(self, *chains: Sequential) -> None:
        super().__init__()
        self.chains = nn.ModuleList(chains)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        chain_outputs = [f(x) for f in self.chains]
        result = chain_outputs[0]
        for y in chain_outputs[1:]:
            result += y
        result = self.act(result)

        return result


class ReductionBlockConcat(nn.Module):
    """Reduction block that concatenates its inputs"""

    def __init__(self, *chains: Sequential) -> None:
        super().__init__()
        self.chains = chains

    def forward(self, x: Tensor) -> Tensor:
        chain_outputs = [f(x) for f in self.chains]
        return torch.cat(chain_outputs, 1)
