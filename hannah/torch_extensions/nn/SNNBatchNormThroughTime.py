import torch


class BatchNormalizationThroughTime1D(torch.nn.Module):
    def __init__(
        self,
        channels,
        timesteps: int = 0,
        eps: float = 1e-4,
        momentum: float = 0.1,
        variant="BNTTv1",
    ):
        super(BatchNormalizationThroughTime1D, self).__init__()
        self.variant = variant
        self.bnttlayer = torch.nn.ModuleList()
        for _ in range(timesteps):
            self.bnttlayer.append(
                torch.nn.BatchNorm1d(channels, eps=eps, momentum=momentum)
            )

    def forward(self, x):
        timesteps = x.shape[2]
        new = x.clone()
        for t in range(timesteps):
            if self.variant == "BNTTv1" or x.shape[0] == 1:
                new[:, :, t] = self.bnttlayer[t](x)[:, :, t]
            elif self.variant == "BNTTv2":
                new[:, :, t] = self.bnttlayer[t](x[:, :, t])
        x = new
        return x
