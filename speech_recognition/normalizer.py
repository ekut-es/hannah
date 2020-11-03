import torch.nn as nn


class FixedPointNormalizer(nn.Module):
    "Simple feature normalizer for fixed point models"

    def __init__(self, normalize_bits: int = 8, normalize_max: int = 256):
        self.normalize_bits = normalize_bits
        self.normalize_max = normalize_max

    def forward(self, x):
        normalize_factor = 2.0 ** (self.normalize_bits - 1)

        x = x * normalize_factor / self.normalize_max
        x = x.round()
        x = x / normalize_factor
        x = x.clamp(-1.0, 1.0 - 1.0 / normalize_factor)

        return x
