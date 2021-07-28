import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single


# Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” Arxiv
# https://github.com/mravanelli/SincNet
class SincConv(nn.Module):
    """Sinc convolution:
    Parameters:
    -----------------
        in_channels: No. of input channels(must be 1)
        out_channels: No. of filters(40)
        sample_rate: sampling rate, default set at 32000
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
        in_channels=1,
        out_channels=40,
        sample_rate=16000,
        kernel_size=101,
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
        self.sample_rate = sample_rate
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
        high_hz = sample_rate / 2 - (self.min_band_hz + self.min_low_hz)

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

        n_ = 2 * math.pi * torch.arange(-N, 0).view(1, -1) / self.sample_rate
        self.register_buffer("n_", n_, persistent=False)

    def forward(self, waveforms):
        f_low = torch.abs(self.low_freq_) + self.min_low_hz
        f_high = torch.clamp(
            f_low + self.min_band_hz + torch.abs(self.band_freq_),
            self.min_low_hz,
            self.sample_rate / 2,
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

        result = F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )

        return result


class Sinc_Act(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):

        return torch.log10(torch.abs(input) + 1)


class SincConvBlock(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=40,
        sample_rate=16000,
        kernel_size=101,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        min_low_hz=50,
        min_band_hz=50,
        bn_len=40,
        pool_len=20,
    ):

        super(SincConvBlock, self).__init__()

        self.layer = nn.Sequential(
            SincConv(
                in_channels=in_channels,
                out_channels=out_channels,
                sample_rate=sample_rate,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding if padding else kernel_size // 2,
                dilation=dilation,
                bias=bias,
                groups=groups,
                min_low_hz=min_low_hz,
                min_band_hz=min_band_hz,
            ),
            Sinc_Act(),
            nn.BatchNorm1d(bn_len),
            nn.AvgPool1d(pool_len),
        )

    def forward(self, x):

        out = self.layer(x)

        return out


class RawFeatures(nn.Module):
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        return x


class SincConvFFT(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self, out_channels=40, sample_rate=16000, min_low_hz=50, min_band_hz=50
    ):
        super(SincConvFFT, self).__init__()

        self.out_channels = out_channels
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initializing filter banks in audible frequency range
        low_hz = self.min_low_hz
        high_hz = sample_rate / 2 - (self.min_band_hz + self.min_low_hz)

        mel = np.linspace(
            self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1
        )
        hz = self.to_hz(mel)

        self.low_freq_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_freq_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

    def forward(self, x):
        spectrogram = torch.stft(x.squeeze(1), n_fft=160, hop_length=160)
        spectrogram = spectrogram[:, :, :, 0]

        f_low = torch.abs(self.low_freq_) + self.min_low_hz
        f_high = torch.clamp(
            f_low + self.min_band_hz + torch.abs(self.band_freq_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band_width = self.sample_rate / (2 * spectrogram.shape[1])
        f_low_idx = f_low.div(band_width).floor()
        f_high_idx = f_high.div(band_width).ceil()
        channels = [
            spectrogram[:, f_low_idx.int()[i] : f_high_idx.int()[i], :].sum(dim=1)
            for i in range(self.out_channels)
        ]
        sinc_test_features = torch.stack(channels, dim=1)
        sinc_test_features = sinc_test_features.div(sinc_test_features.max())

        return sinc_test_features
