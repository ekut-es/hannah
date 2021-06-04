import torch


class Spiking1DIFLayer(torch.nn.Module):
    def __init__(self, channels: int, spike_fn, flatten_output=False, time_position=2):

        super(Spiking1DIFLayer, self).__init__()

        self.channels = channels
        self.spike_fn = spike_fn
        self.flatten_output = flatten_output

        self.Vth = torch.ones(channels)
        self.time_position = time_position

        self.type = "IF"

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):

        self.Vth = self.Vth.to(device=x.device)

        batch_size = x.shape[0]
        nb_steps = x.shape[2]

        # membrane potential
        mem = torch.zeros((batch_size, self.channels), dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.channels), dtype=x.dtype, device=x.device)
        if self.time_position == 2:
            # output spikes recording
            spk_rec = torch.zeros(
                (batch_size, self.channels, nb_steps), dtype=x.dtype, device=x.device
            )
        elif self.time_position == 1:
            # output spikes recording
            spk_rec = torch.zeros(
                (batch_size, nb_steps, self.channels), dtype=x.dtype, device=x.device
            )
        for t in range(nb_steps):
            rst = torch.einsum("ab,b->ab", spk, self.Vth)

            if self.time_position == 2:
                input_ = x[:, :, t]
            elif self.time_position == 1:
                input_ = x[:, t, :]

            mem = (mem - rst) + input_

            spk = self.spike_fn(mem - self.Vth)

            if self.time_position == 2:
                spk_rec[:, :, t] = spk
            elif self.time_position == 1:
                spk_rec[:, t, :] = spk

        if self.flatten_output:
            output = torch.transpose(spk_rec, 1, 2).contiguous()
        else:
            output = spk_rec

        return output


class Spiking1DeLIFLayer(torch.nn.Module):
    def __init__(
        self, channels: int, spike_fn, flatten_output=False, beta=0.65, time_position=2
    ):

        super(Spiking1DeLIFLayer, self).__init__()

        self.channels = channels
        self.spike_fn = spike_fn
        self.flatten_output = flatten_output

        self.beta = torch.tensor(beta)
        self.Vth = torch.ones(channels)

        self.type = "eLIF"
        self.time_position = time_position

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):

        self.beta = self.beta.to(device=x.device)
        self.Vth = self.Vth.to(device=x.device)

        batch_size = x.shape[0]
        nb_steps = x.shape[self.time_position]

        # membrane potential
        mem = torch.zeros((batch_size, self.channels), dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.channels), dtype=x.dtype, device=x.device)
        if self.time_position == 2:
            # output spikes recording
            spk_rec = torch.zeros(
                (batch_size, self.channels, nb_steps), dtype=x.dtype, device=x.device
            )
        elif self.time_position == 1:
            # output spikes recording
            spk_rec = torch.zeros(
                (batch_size, nb_steps, self.channels), dtype=x.dtype, device=x.device
            )

        for t in range(nb_steps):
            rst = torch.einsum("ab,b->ab", spk, self.Vth)

            if self.time_position == 2:
                input_ = x[:, :, t]
            elif self.time_position == 1:
                input_ = x[:, t, :]

            mem = (mem - rst) * self.beta + input_

            spk = self.spike_fn(mem - self.Vth)
            if self.time_position == 2:
                spk_rec[:, :, t] = spk
            elif self.time_position == 1:
                spk_rec[:, t, :] = spk

        if self.flatten_output:
            output = torch.transpose(spk_rec, 1, 2).contiguous()
        else:
            output = spk_rec

        return output

    def reset_parameters(self):
        pass

    def clamp(self):
        pass


class Spiking1DLIFLayer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        spike_fn,
        flatten_output=False,
        alpha=0.75,
        beta=0.65,
        time_position=2,
    ):

        super(Spiking1DLIFLayer, self).__init__()

        self.channels = channels
        self.spike_fn = spike_fn
        self.flatten_output = flatten_output

        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.beta = torch.nn.Parameter(torch.tensor(beta), requires_grad=False)
        self.Vth = torch.nn.Parameter(torch.ones(channels), requires_grad=False)

        self.type = "LIF"
        self.time_position = time_position

        self.training = True

    def forward(self, x):

        self.alpha = self.alpha.to(device=x.device)
        self.beta = self.beta.to(device=x.device)
        self.Vth = self.Vth.to(device=x.device)

        batch_size = x.shape[0]
        nb_steps = x.shape[2]

        # membrane potential
        mem = torch.zeros((batch_size, self.channels), dtype=x.dtype, device=x.device)

        # input current
        input_ = torch.zeros(
            (batch_size, self.channels), dtype=x.dtype, device=x.device
        )

        # output spikes
        spk = torch.zeros((batch_size, self.channels), dtype=x.dtype, device=x.device)
        if self.time_position == 2:
            # output spikes recording
            spk_rec = torch.zeros(
                (batch_size, self.channels, nb_steps), dtype=x.dtype, device=x.device
            )
        elif self.time_position == 1:
            # output spikes recording
            spk_rec = torch.zeros(
                (batch_size, nb_steps, self.channels), dtype=x.dtype, device=x.device
            )

        for t in range(nb_steps):
            rst = torch.einsum("ab,b->ab", spk, self.Vth)

            if self.time_position == 2:
                input_ = x[:, :, t]
            elif self.time_position == 1:
                input_ = x[:, t, :]

            mem = (mem - rst) * self.beta + input_

            spk = self.spike_fn(mem - self.Vth)

            if self.time_position == 2:
                spk_rec[:, :, t] = spk
            elif self.time_position == 1:
                spk_rec[:, t, :] = spk

        if self.flatten_output:
            output = torch.transpose(spk_rec, 1, 2).contiguous()
        else:
            output = spk_rec

        return output


class Spiking1DeALIFLayer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        spike_fn,
        flatten_output=False,
        beta=0.65,
        gamma=0.75,
        rho=0.75,
        time_position=2,
    ):

        super(Spiking1DeALIFLayer, self).__init__()

        self.channels = channels
        self.spike_fn = spike_fn
        self.flatten_output = flatten_output

        self.beta = torch.tensor(beta)
        self.gamma = torch.tensor(gamma)
        self.rho = torch.tensor(rho)
        self.Vth = torch.ones(channels)

        self.type = "eALIF"
        self.time_position = time_position

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):

        self.beta = self.beta.to(device=x.device)
        self.gamma = self.gamma.to(device=x.device)
        self.Vth = self.Vth.to(device=x.device)

        batch_size = x.shape[0]
        nb_steps = x.shape[2]

        # membrane potential
        mem = torch.zeros((batch_size, self.channels), dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.channels), dtype=x.dtype, device=x.device)
        if self.time_position == 2:
            # output spikes recording
            spk_rec = torch.zeros(
                (batch_size, self.channels, nb_steps), dtype=x.dtype, device=x.device
            )
        elif self.time_position == 1:
            # output spikes recording
            spk_rec = torch.zeros(
                (batch_size, nb_steps, self.channels), dtype=x.dtype, device=x.device
            )

        Athpot = torch.ones((batch_size, self.channels), dtype=x.dtype, device=x.device)

        thadapt = torch.zeros(
            (batch_size, self.channels), dtype=x.dtype, device=x.device
        )

        for t in range(nb_steps):
            rst = torch.einsum("ab,b->ab", spk, self.Vth)

            if self.time_position == 2:
                input_ = x[:, :, t]
            elif self.time_position == 1:
                input_ = x[:, t, :]

            mem = (mem - rst) * self.beta + input_

            thadapt = self.rho * thadapt + spk_rec[:, :, t - 1]

            Athpot = self.Vth + self.gamma * thadapt

            spk = self.spike_fn(mem - Athpot)

            if self.time_position == 2:
                spk_rec[:, :, t] = spk
            elif self.time_position == 1:
                spk_rec[:, t, :] = spk

        if self.flatten_output:
            output = torch.transpose(spk_rec, 1, 2).contiguous()
        else:
            output = spk_rec

        return output


class Spiking1DALIFLayer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        spike_fn,
        flatten_output=False,
        alpha=0.75,
        beta=0.65,
        gamma=0.75,
        rho=0.75,
        time_position=2,
    ):

        super(Spiking1DALIFLayer, self).__init__()

        self.channels = channels
        self.spike_fn = spike_fn
        self.flatten_output = flatten_output

        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.gamma = torch.tensor(gamma)
        self.rho = torch.tensor(rho)
        self.Vth = torch.ones(channels)

        self.type = "ALIF"
        self.time_position = time_position

        self.training = True

    def forward(self, x):

        self.alpha = self.alpha.to(device=x.device)
        self.beta = self.beta.to(device=x.device)
        self.gamma = self.gamma.to(device=x.device)
        self.Vth = self.Vth.to(device=x.device)

        batch_size = x.shape[0]
        nb_steps = x.shape[2]

        # membrane potential
        mem = torch.zeros((batch_size, self.channels), dtype=x.dtype, device=x.device)
        # input current
        input_ = torch.zeros(
            (batch_size, self.channels), dtype=x.dtype, device=x.device
        )
        # output spikes
        spk = torch.zeros((batch_size, self.channels), dtype=x.dtype, device=x.device)
        if self.time_position == 2:
            # output spikes recording
            spk_rec = torch.zeros(
                (batch_size, self.channels, nb_steps), dtype=x.dtype, device=x.device
            )
        elif self.time_position == 1:
            # output spikes recording
            spk_rec = torch.zeros(
                (batch_size, nb_steps, self.channels), dtype=x.dtype, device=x.device
            )

        Athpot = torch.ones((batch_size, self.channels), dtype=x.dtype, device=x.device)

        thadapt = torch.zeros(
            (batch_size, self.channels), dtype=x.dtype, device=x.device
        )

        for t in range(nb_steps):
            rst = torch.einsum("ab,b->ab", spk, self.Vth)

            if self.time_position == 2:
                input_ = self.alpha * input_ + x[:, :, t]
            elif self.time_position == 1:
                input_ = self.alpha * input_ + x[:, t, :]

            mem = (mem - rst) * self.beta + input_

            thadapt = self.rho * thadapt + spk_rec[:, :, t - 1]

            Athpot = self.Vth + self.gamma * thadapt

            spk = self.spike_fn(mem - Athpot)

            if self.time_position == 2:
                spk_rec[:, :, t] = spk
            elif self.time_position == 1:
                spk_rec[:, t, :] = spk

        if self.flatten_output:
            output = torch.transpose(spk_rec, 1, 2).contiguous()
        else:
            output = spk_rec

        return output


class ActivationLayer(torch.nn.Module):
    def __init__(self, act=None):
        assert act is not None, "Activation not set"
        super(ActivationLayer, self).__init__()
        self.act = act

    def forward(self, x):
        return self.act(x)


class Surrogate_BP_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input
            * 0.3
            * torch.nn.functional.threshold(1.0 - torch.abs(input), 0, 0)
        )
        return grad


class SurrogateHeaviside(torch.autograd.Function):

    # Activation function with surrogate gradient
    sigma = 10.0

    @staticmethod
    def forward(ctx, input):

        output = torch.zeros_like(input)
        output[input > 0] = 1.0
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # approximation of the gradient using sigmoid function
        grad = (
            grad_input
            * torch.sigmoid(SurrogateHeaviside.sigma * input)
            * torch.sigmoid(-SurrogateHeaviside.sigma * input)
        )
        return grad
