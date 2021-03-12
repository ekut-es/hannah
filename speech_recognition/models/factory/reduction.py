import torch
import torch.nn as nn


class ReductionBlockAdd(nn.Module):
    """Reduction block that sums over its inputs"""

    def __init__(self, *chains):
        super().__init__()
        self.chains = nn.ModuleList(chains)
        self.act = nn.ReLU()

    def forward(self, x):
        chain_outputs = [f(x) for f in self.chains]
        result = torch.zeros_like(chain_outputs[0])
        for out in chain_outputs:
            result += out

        result = self.act(result)

        return result


class ReductionBlockConcat(nn.Module):
    """Reduction block that concatenates its inputs"""

    def __init__(self, *chains):
        super().__init__()
        self.chains = chains

    def forward(self, x):
        chain_outputs = [f(x) for f in self.chains]
        return torch.cat(chain_outputs, 1)
