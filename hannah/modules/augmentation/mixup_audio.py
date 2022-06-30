import numpy.random as random
import torch
import torch.nn as nn


class MixupAudio(nn.Module):
    "An implementation of CutMix and MixUp for audio Spectrograms"

    def __init__(self, n_labels, spec_mix_prob=0.5, prob=1.0, alpha=0.3, seed=1234):
        super().__init__()
        self.n_labels = n_labels
        self.spec_mix_prob = spec_mix_prob
        self.alpha = alpha
        self.prob = prob

        self.random_state = random.RandomState(seed=seed)

    def _mixup(self, x: torch.Tensor, y: torch.Tensor):
        batch_size = x.size(0)

        lam = self.random_state.beta(self.alpha, self.alpha)
        permutation = torch.tensor(
            self.random_state.permutation(batch_size), device=x.device
        )

        x_permuted = x[permutation]
        y_permuted = y[permutation]

        x = (1 - lam) * x + lam * x_permuted
        y = (1 - lam) * y + lam * y_permuted

        return x, y

    def _spec_mix(self, x, y):
        batch_size = x.size(0)

        lam = self.random_state.beta(self.alpha, self.alpha)
        permutation = torch.tensor(
            self.random_state.permutation(batch_size), device=x.device
        )

        x_permuted = x[permutation]
        y_permuted = y[permutation]

        channels = x.size(1)
        lam_channels = int(round(channels * lam))

        lam = lam_channels / channels

        for batch in range(batch_size):
            channel_choices = torch.tensor(
                sorted(self.random_state.choice(channels, lam_channels, replace=False)),
                device=x.device,
                dtype=torch.int64,
            )

            x[batch][channel_choices] = x_permuted[batch][channel_choices]

        y = (1 - lam) * y + lam * y_permuted

        return x, y

    def forward(self, x, y):
        assert x.shape[0] == y.shape[0]

        if y.dim() == 2 and y.size(1) == 1:
            y = y.squeeze()
        if y.dim() == 1:
            y = nn.functional.one_hot(y.long()).float()

        if self.random_state.uniform() <= self.prob:
            if self.random_state.uniform() <= self.spec_mix_prob:
                x, y = self._spec_mix(x, y)
            else:
                x, y = self._mixup(x, y)

        return x, y
