import numpy as np
import torch


class ReadoutLayer(torch.nn.Module):

    "Fully connected readout"

    def __init__(self, input_shape, output_shape, w_init_mean, w_init_std, eps=1e-8):
        super(ReadoutLayer, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.eps = eps

        self.w = torch.nn.Parameter(
            torch.empty((input_shape, output_shape)), requires_grad=True
        )
        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)

        self.reset_parameters()

        self.mem_rec_hist = None

    def forward(self, x):
        h = torch.einsum("abc,cd->abd", x, self.w)

        norm = (self.w**2).sum(0)

        mem_rec = h
        output = torch.mean(mem_rec, 1) / (norm + 1e-8) - self.b

        return output

    def reset_parameters(self):
        torch.nn.init.normal_(
            self.w,
            mean=self.w_init_mean,
            std=self.w_init_std * np.sqrt(1.0 / (self.input_shape)),
        )

        torch.nn.init.normal_(self.b, mean=1.0, std=0.01)


class ReadoutMeanLayer(torch.nn.Module):

    "Fully connected readout"

    def __init__(self, output_shape, trainable_parameter=False):
        super(ReadoutMeanLayer, self).__init__()

        self.output_shape = output_shape
        self.b = torch.nn.Parameter(torch.ones(output_shape), requires_grad=False)

    def forward(self, x):
        mem_rec = x
        output = torch.mean(mem_rec, 1) - self.b

        return output

    def reset_parameters(self):
        if self.trainable_parameter:
            torch.nn.init.normal_(self.alpha, mean=0.7, std=0.01)
            torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
            torch.nn.init.normal_(self.gamma, mean=0.7, std=0.01)
            torch.nn.init.normal_(self.rho, mean=0.7, std=0.01)

    def clamp(self):
        if self.trainable_parameter:
            self.alpha.data.clamp_(0.0, 1.0)
            self.beta.data.clamp_(0.0, 1.0)
            self.gamma.data.clamp_(0.0, 1.0)
            self.rho.data.clamp_(0.0, 1.0)


class ReadoutSpikeTimeLayer(torch.nn.Module):
    def __init__(
        self,
    ):

        super(ReadoutSpikeTimeLayer, self).__init__()

    def forward(self, x):
        nb_steps = x.shape[1]
        batch_size = x.shape[0]
        channels = x.shape[2]
        output = torch.zeros((batch_size, channels), dtype=x.dtype, device=x.device)
        for t in range(nb_steps):
            output[:, :] += x[:, t, :] * (nb_steps - 1 - t)
        return output


class ReadoutCountLayer(torch.nn.Module):
    def __init__(
        self,
    ):

        super(ReadoutCountLayer, self).__init__()

    def forward(self, x):
        nb_steps = x.shape[1]
        batch_size = x.shape[0]
        channels = x.shape[2]
        output = torch.zeros((batch_size, channels), dtype=x.dtype, device=x.device)
        for t in range(nb_steps):
            output[:, :] += x[:, t, :]
        return output


class ReadoutFirstSpikeLayer(torch.nn.Module):
    def __init__(
        self,
    ):

        super(ReadoutFirstSpikeLayer, self).__init__()

    def forward(self, x):
        nb_steps = x.shape[1]
        batch_size = x.shape[0]
        output = torch.zeros((batch_size, x.shape[2]), dtype=x.dtype, device=x.device)
        for b in range(batch_size):
            for t in range(nb_steps):
                if torch.max(x[b, t, :]).item() == 1:
                    for i in range(x.shape[2]):
                        if x[b, t, i] == 1 and output[b, i] < ((nb_steps - 1) - t):
                            output[b, i] = (nb_steps - 1) - t
        return output
