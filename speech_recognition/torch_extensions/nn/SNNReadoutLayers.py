import numpy as np
import torch


class ReadoutLayer(torch.nn.Module):

    "Fully connected readout"

    def __init__(
        self,
        input_shape,
        output_shape,
        w_init_mean,
        w_init_std,
        eps=1e-8,
        time_reduction="mean",
    ):

        assert time_reduction in [
            "mean",
            "max",
        ], 'time_reduction should be "mean" or "max"'

        super(ReadoutLayer, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.eps = eps
        self.time_reduction = time_reduction

        self.w = torch.nn.Parameter(
            torch.empty((input_shape, output_shape)), requires_grad=True
        )
        if time_reduction == "max":
            self.beta = torch.nn.Parameter(
                torch.tensor(0.7 * np.ones((1))), requires_grad=True
            )
        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)

        self.reset_parameters()
        self.clamp()

        self.mem_rec_hist = None

    def forward(self, x):

        batch_size = x[0].shape[0]

        h = torch.einsum("abc,cd->abd", x, self.w)

        norm = (self.w ** 2).sum(0)

        if self.time_reduction == "max":
            nb_steps = x.shape[1]
            # membrane potential
            mem = torch.zeros(
                (batch_size, self.output_shape), dtype=x.dtype, device=x.device
            )

            # memrane potential recording
            mem_rec = torch.zeros(
                (batch_size, nb_steps, self.output_shape),
                dtype=x.dtype,
                device=x.device,
            )

            for t in range(nb_steps):

                # membrane potential update
                mem = mem * self.beta + (1 - self.beta) * h[:, t, :]
                mem_rec[:, t, :] = mem

            output = torch.max(mem_rec, 1)[0] / (norm + 1e-8) - self.b

        elif self.time_reduction == "mean":

            mem_rec = h
            output = torch.mean(mem_rec, 1) / (norm + 1e-8) - self.b

        return output


class ReadoutMeanLayer(torch.nn.Module):

    "Fully connected readout"

    def __init__(self, output_shape):
        super(ReadoutMeanLayer, self).__init__()

        self.output_shape = output_shape
        self.b = torch.nn.Parameter(torch.ones(output_shape), requires_grad=False)

    def forward(self, x):
        mem_rec = x
        output = torch.mean(mem_rec, 1) - self.b

        return output


class ReadoutSpikeTimeLayer(torch.nn.Module):
    def __init__(self,):

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
    def __init__(self,):

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
    def __init__(self,):

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
