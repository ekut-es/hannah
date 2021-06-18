import pytest

import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram, MFCC

from hannah.models.factory.tracer import QuantizationTracer, RelayConverter
from hannah.models.factory.qat import ConvBn1d, ConvBn2d, ConvBnReLU1d, ConvBnReLU2d
from hannah.models.factory.qconfig import get_trax_qat_qconfig


class Config:
    bw_b = 8
    bw_f = 8
    bw_w = 6

    def get(self, name: str, default=None):
        return getattr(self, name, default)


class TestCell(nn.Module):
    def __init__(self, dim=1, act=False):
        super().__init__()
        if dim == 1:
            if act:
                self.conv = ConvBn1d(8, 8, 3, qconfig=get_trax_qat_qconfig(Config()))
            else:
                self.conv = ConvBnReLU1d(
                    8, 8, 3, qconfig=get_trax_qat_qconfig(Config())
                )
        elif dim == 2:
            if act:
                self.conv = ConvBnReLU2d(
                    8, 8, 3, qconfig=get_trax_qat_qconfig(Config())
                )
            else:
                self.conv = ConvBn2d(8, 8, 3, qconfig=get_trax_qat_qconfig(Config()))

    def forward(self, x):
        return self.conv(x)


@pytest.mark.parametrize("dim,act", [(1, False), (1, True), (2, False), (2, True)])
def test_tracer(dim, act):
    cell = TestCell(dim=dim, act=act)
    tracer = QuantizationTracer()

    traced_graph = tracer.trace(cell)

    converter = RelayConverter(torch.fx.GraphModule(cell, traced_graph))
    if dim == 1:
        input = torch.rand((1, 8, 12))
    elif dim == 2:
        input = torch.rand((1, 8, 12, 12))
    converter.run(input)


if __name__ == "__main__":
    test_tracer(1, False)
    test_tracer(2, True)
