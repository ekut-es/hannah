import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram, MFCC

from speech_recognition.models.factory.tracer import QuantizationTracer, RelayConverter
from speech_recognition.models.factory.qat import ConvBn1d
from speech_recognition.models.factory.qconfig import get_trax_qat_qconfig


class Config:
    bw_b = 8
    bw_f = 8
    bw_w = 6

    def get(self, name: str, default=None):
        return getattr(self, name, default)


class TestCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBn1d(8, 8, 3, qconfig=get_trax_qat_qconfig(Config()))

    def forward(self, x):
        return self.conv(x)


def test_tracer():
    cell = TestCell()
    tracer = QuantizationTracer()

    traced_graph = tracer.trace(cell)
    print(traced_graph)

    converter = RelayConverter(torch.fx.GraphModule(cell, traced_graph))
    input = torch.rand((1, 8, 12))
    converter.run(input)


if __name__ == "__main__":
    test_tracer()
