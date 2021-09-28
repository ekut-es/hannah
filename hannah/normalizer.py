from typing import Optional

import torch
import torch.nn as nn


class FixedPointNormalizer(nn.Module):
    "Simple feature normalizer for fixed point models"

    def __init__(self, normalize_bits: int = 8, normalize_max: int = 256):
        super().__init__()
        self.normalize_bits = normalize_bits
        self.normalize_max = normalize_max

    def forward(self, x):
        normalize_factor = 2.0 ** (self.normalize_bits - 1)
        x = x * normalize_factor / self.normalize_max
        x = x.round()
        x = x / normalize_factor
        x = x.clamp(-1.0, 1.0 - 1.0 / normalize_factor)

        return x


class AdaptiveFixedPointNormalizer(nn.Module):
    "Simple feature normalizer for fixed point models"

    def __init__(self, normalize_bits: int = 8, normalize_max: int = 256):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=40, affine=False)

    def forward(self, x):
        # print(torch.max(self.bn.running_mean))
        # print(torch.max(self.bn.running_var))
        return self.bn(x)


class HistogramNormalizer(nn.Module):
    def __init__(self, bits: int = 8, bins: Optional[int] = None):
        super().__init__()

        self.bits = bits
        if bins is None:
            bins = min(2 ** bits, 2048)
        self.bins = bins

        self.register_buffer("histogram", torch.zeros(self.bins))
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.detach()

        new_min = torch.min(x)
        new_max = torch.max(x)

        self.min_val = torch.min(self.min_val, new_min)
        self.max_val = torch.max(self.max_val, new_max)

        histogram = torch.histc(
            x, min=int(self.min_val), max=int(self.max_val), bins=self.bins
        )

        print(self.min_val)
        print(self.max_val)

        return x_orig
