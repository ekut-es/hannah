import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, convolutions, pooling, linear, dropout, qconfig=None):
        super().__init__()
        self.convolutions = convolutions
        self.pooling = pooling
        self.dropout = dropout
        self.flatten = nn.Flatten()
        self.linear = linear
        self.qconfig = qconfig
        if qconfig:
            self.activation_post_process = self.qconfig.activation()

    def forward(self, x):
        if hasattr(self, "activation_post_process"):
            x = self.activation_post_process(x)
        x = self.convolutions(x)
        x = self.pooling(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
