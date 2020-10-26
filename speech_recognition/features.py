import torch.nn as nn
import numpy as np


# Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” Arxiv
# https://github.com/mravanelli/SincNet


class SincConv(nn.Module):
    """Sinc convolution:
        Parameters:
        -----------------
            in_channels: No. of input channels(must be 1)
            out_channels: No. of filters(40)
            SR: sampling rate, default set at 32000
            kernel_size: Filter length(101)
            """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        SR=16000,
        in_channels=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        min_low_hz=50,
        min_band_hz=50,
    ):
        super(SincConv, self).__init__()

        if in_channels != 1:
            err = "SincConv only suports one input channel."
            raise ValueError(err)

        if bias:
            raise ValueError("SincConv does not support bias.")

        if groups > 1:
            raise ValueError("SincConv only supports one group.")

        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.SR = SR
        self.in_channels = in_channels
        self.groups = groups
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        if self.kernel_size[0] % 2 == 0:
            self.kernel_size[0] = (
                self.kernel_size[0] + 1
            )  # odd length so that filter is symmetric

        # initializing filter banks in audible frequency range
        low_hz = self.min_low_hz
        high_hz = SR / 2 - (self.min_band_hz + self.min_low_hz)

        mel = np.linspace(
            self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1
        )
        hz = self.to_hz(mel)

        self.low_freq_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_freq_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # hamming window
        N = (self.kernel_size[0] - 1) / 2.0
        window_ = torch.hamming_window(self.kernel_size[0])
        self.register_buffer("window_", window_, persistent=False)

        n_ = 2 * math.pi * torch.arange(-N, 0).view(1, -1) / self.SR
        self.register_buffer("n_", n_, persistent=False)

    def forward(self, waveforms):

        f_low = torch.abs(self.low_freq_) + self.min_low_hz
        f_high = torch.clamp(
            f_low + self.min_band_hz + torch.abs(self.band_freq_),
            self.min_low_hz,
            self.SR / 2,
        )
        f_band = (f_high - f_low)[:, 0]

        f_n_low = torch.matmul(f_low, self.n_)
        f_n_high = torch.matmul(f_high, self.n_)

        bpl = (torch.sin(f_n_high) - torch.sin(f_n_low)) / (
            self.n_ / 2
        )  # *self.window_
        bpr = torch.flip(bpl, dims=[1])
        bpc = 2 * f_band.view(-1, 1)

        band = torch.cat([bpl, bpc, bpr], dim=1)
        band = band / (2 * f_band[:, None])
        band = band * self.window_[None,]

        self.filters = band.view(self.out_channels, 1, self.kernel_size[0])

        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )
