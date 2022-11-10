#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import torch

# Taken from Paper Low-activity supervised convolutional spiking neural networks applied to speech commands recognition Arxiv:2011.06846


class SNN(torch.nn.Module):
    def __init__(self, layers):

        super(SNN, self).__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        loss_seq = []

        for l in self.layers:
            x, loss = l(x)
            loss_seq.append(loss)

        return x, loss_seq

    def clamp(self):

        for l in self.layers:
            l.clamp()

    def reset_parameters(self):

        for l in self.layers:
            l.reset_parameters()


class EmptyLayer(torch.nn.Module):
    def __init__(
        self,
    ):

        super(EmptyLayer, self).__init__()

    def forward(self, x):

        return x


class TimeTransposeLayer(torch.nn.Module):
    def __init__(self, time_position=2):

        super(TimeTransposeLayer, self).__init__()
        self.time_position = time_position

    def forward(self, x):
        if self.time_position == 1:
            output = torch.transpose(x, 1, 2).contiguous()
        elif self.time_position == 2:
            output = torch.transpose(x, 2, 1).contiguous()

        return output


class SpikingDenseLayer(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        spike_fn,
        w_init_mean,
        w_init_std,
        recurrent=False,
        lateral_connections=True,
        eps=1e-8,
    ):

        super(SpikingDenseLayer, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.eps = eps
        self.lateral_connections = lateral_connections

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.w = torch.nn.Parameter(
            torch.empty((input_shape, output_shape)), requires_grad=True
        )
        if recurrent:
            self.v = torch.nn.Parameter(
                torch.empty((output_shape, output_shape)), requires_grad=True
            )

        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):

        batch_size = x.shape[0]

        h = torch.einsum("abc,cd->abd", x, self.w)
        nb_steps = h.shape[1]

        # membrane potential
        mem = torch.zeros(
            (batch_size, self.output_shape), dtype=x.dtype, device=x.device
        )
        # output spikes
        spk = torch.zeros(
            (batch_size, self.output_shape), dtype=x.dtype, device=x.device
        )

        # output spikes recording
        spk_rec = torch.zeros(
            (batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device
        )

        if self.lateral_connections:
            d = torch.einsum("ab, ac -> bc", self.w, self.w)

        norm = (self.w**2).sum(0)

        for t in range(nb_steps):

            # reset term
            if self.lateral_connections:
                rst = torch.einsum("ab,bc ->ac", spk, d)
            else:
                rst = spk * self.b * norm

            input_ = h[:, t, :]
            if self.recurrent:
                input_ = input_ + torch.einsum("ab,bc->ac", spk, self.v)

            # membrane potential update
            mem = (mem - rst) * self.beta + input_ * (1.0 - self.beta)
            mthr = torch.einsum("ab,b->ab", mem, 1.0 / (norm + self.eps)) - self.b

            spk = self.spike_fn(mthr)

            spk_rec[:, t, :] = spk

        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        return spk_rec

    def reset_parameters(self):

        torch.nn.init.normal_(
            self.w,
            mean=self.w_init_mean,
            std=self.w_init_std * np.sqrt(1.0 / self.input_shape),
        )
        if self.recurrent:
            torch.nn.init.normal_(
                self.v,
                mean=self.w_init_mean,
                std=self.w_init_std * np.sqrt(1.0 / self.output_shape),
            )
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1.0, std=0.01)

    def clamp(self):

        self.beta.data.clamp_(0.0, 1.0)
        self.b.data.clamp_(min=0.0)


class SpikingConv2DLayer(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        spike_fn,
        w_init_mean,
        w_init_std,
        recurrent=False,
        lateral_connections=True,
        eps=1e-8,
        stride=(1, 1),
        flatten_output=False,
    ):

        super(SpikingConv2DLayer, self).__init__()

        self.kernel_size = np.array(kernel_size)
        self.dilation = np.array(dilation)
        self.stride = np.array(stride)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.lateral_connections = lateral_connections
        self.eps = eps

        self.flatten_output = flatten_output

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.w = torch.nn.Parameter(
            torch.empty((out_channels, in_channels, *kernel_size)), requires_grad=True
        )
        if recurrent:
            self.v = torch.nn.Parameter(
                torch.empty((out_channels, out_channels)), requires_grad=True
            )
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=True)

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):

        batch_size = x.shape[0]

        conv_x = torch.nn.functional.conv2d(
            x,
            self.w,
            padding=tuple(
                np.ceil(((self.kernel_size - 1) * self.dilation) / 2).astype(int)
            ),
            dilation=tuple(self.dilation),
            stride=tuple(self.stride),
        )
        conv_x = conv_x[:, :, :, : self.output_shape]
        nb_steps = conv_x.shape[2]

        # membrane potential
        mem = torch.zeros(
            (batch_size, self.out_channels, self.output_shape),
            dtype=x.dtype,
            device=x.device,
        )
        # output spikes
        spk = torch.zeros(
            (batch_size, self.out_channels, self.output_shape),
            dtype=x.dtype,
            device=x.device,
        )

        # output spikes recording
        spk_rec = torch.zeros(
            (batch_size, self.out_channels, nb_steps, self.output_shape),
            dtype=x.dtype,
            device=x.device,
        )

        if self.lateral_connections:
            d = torch.einsum("abcd, ebcd -> ae", self.w, self.w)
        b = self.b.unsqueeze(1).repeat((1, self.output_shape))

        norm = (self.w**2).sum((1, 2, 3))

        for t in range(nb_steps):

            # reset term
            if self.lateral_connections:
                rst = torch.einsum("abc,bd ->adc", spk, d)
            else:
                rst = torch.einsum("abc,b,b->abc", spk, self.b, norm)

            input_ = conv_x[:, :, t, :]
            if self.recurrent:
                input_ = input_ + torch.einsum("abc,bd->adc", spk, self.v)

            # membrane potential update
            mem = (mem - rst) * self.beta + input_ * (1.0 - self.beta)
            mthr = torch.einsum("abc,b->abc", mem, 1.0 / (norm + self.eps)) - b

            spk = self.spike_fn(mthr)

            spk_rec[:, :, t, :] = spk

        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        if self.flatten_output:

            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(
                batch_size, nb_steps, self.out_channels * self.output_shape
            )

        else:

            output = spk_rec

        return output

    def reset_parameters(self):

        torch.nn.init.normal_(
            self.w,
            mean=self.w_init_mean,
            std=self.w_init_std
            * np.sqrt(1.0 / (self.in_channels * np.prod(self.kernel_size))),
        )
        if self.recurrent:
            torch.nn.init.normal_(
                self.v,
                mean=self.w_init_mean,
                std=self.w_init_std * np.sqrt(1.0 / self.out_channels),
            )
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1.0, std=0.01)

    def clamp(self):

        self.beta.data.clamp_(0.0, 1.0)
        self.b.data.clamp_(min=0.0)


class SpikingConv3DLayer(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        spike_fn,
        w_init_mean,
        w_init_std,
        recurrent=False,
        lateral_connections=True,
        eps=1e-8,
        stride=(1, 1, 1),
        flatten_output=False,
        negative_mempot=False,
    ):

        super(SpikingConv3DLayer, self).__init__()

        self.kernel_size = np.array(kernel_size)
        self.dilation = np.array(dilation)
        self.stride = np.array(stride)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.lateral_connections = lateral_connections
        self.eps = eps

        self.flatten_output = flatten_output

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.w = torch.nn.Parameter(
            torch.empty((out_channels, in_channels, *kernel_size)), requires_grad=True
        )
        if recurrent:
            self.v = torch.nn.Parameter(
                torch.empty((out_channels, out_channels)), requires_grad=True
            )
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=True)

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None

        self.training = True
        self.negative_mempot = negative_mempot

    def forward(self, x):

        batch_size = x.shape[0]

        conv_x = torch.nn.functional.conv3d(
            x,
            self.w,
            padding=tuple(
                np.ceil(((self.kernel_size - 1) * self.dilation) / 2).astype(int)
            ),
            dilation=tuple(self.dilation),
            stride=tuple(self.stride),
        )
        conv_x = conv_x[:, :, :, : self.output_shape[0], : self.output_shape[1]]
        nb_steps = conv_x.shape[2]

        # membrane potential
        mem = torch.zeros(
            (batch_size, self.out_channels, *self.output_shape),
            dtype=x.dtype,
            device=x.device,
        )
        # output spikes
        spk = torch.zeros(
            (batch_size, self.out_channels, *self.output_shape),
            dtype=x.dtype,
            device=x.device,
        )

        # output spikes recording
        spk_rec = torch.zeros(
            (batch_size, self.out_channels, nb_steps, *self.output_shape),
            dtype=x.dtype,
            device=x.device,
        )

        if self.lateral_connections:
            d = torch.einsum("abcde, fbcde -> af", self.w, self.w)
        b = self.b.unsqueeze(1).unsqueeze(1).repeat((1, *self.output_shape))

        norm = (self.w**2).sum((1, 2, 3, 4))

        for t in range(nb_steps):

            # reset term
            if self.lateral_connections:
                rst = torch.einsum("abcd,be ->aecd", spk, d)
            else:
                rst = torch.einsum("abcd,b,b->abcd", spk, self.b, norm)

            input_ = conv_x[:, :, t, :, :]
            if self.recurrent:
                input_ = input_ + torch.einsum("abcd,be->aecd", spk, self.v)

            # membrane potential update
            if self.negative_mempot:
                mem = mem * self.beta + input_ * (1.0 - self.beta) - rst
            else:
                mem = (mem - rst) * self.beta + input_ * (1.0 - self.beta)
            mthr = torch.einsum("abcd,b->abcd", mem, 1.0 / (norm + self.eps)) - b

            spk = self.spike_fn(mthr)

            spk_rec[:, :, t, :, :] = spk

        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        if self.flatten_output:

            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(
                batch_size, nb_steps, self.out_channels * np.prod(self.output_shape)
            )

        else:

            output = spk_rec

        return output

    def reset_parameters(self):

        torch.nn.init.normal_(
            self.w,
            mean=self.w_init_mean,
            std=self.w_init_std
            * np.sqrt(1.0 / (self.in_channels * np.prod(self.kernel_size))),
        )
        if self.recurrent:
            torch.nn.init.normal_(
                self.v,
                mean=self.w_init_mean,
                std=self.w_init_std * np.sqrt(1.0 / self.out_channels),
            )
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1.0, std=0.01)

    def clamp(self):

        self.beta.data.clamp_(0.0, 1.0)
        self.b.data.clamp_(min=0.0)


class Spiking1DS2NetLayer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        spike_fn,
        eps=1e-8,
        flatten_output=False,
        convolution_layer=None,
    ):

        super(Spiking1DS2NetLayer, self).__init__()

        self.channels = channels
        self.spike_fn = spike_fn
        self.flatten_output = flatten_output
        self.convolution = convolution_layer

        self.eps = eps
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(channels), requires_grad=True)

        self.reset_parameters()
        self.clamp()
        self.type = "s2net"

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):
        batch_size = x.shape[0]
        nb_steps = x.shape[2]

        # membrane potential
        mem = torch.zeros((batch_size, self.channels), dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.channels), dtype=x.dtype, device=x.device)
        # output spikes recording
        spk_rec = torch.zeros(
            (batch_size, self.channels, nb_steps), dtype=x.dtype, device=x.device
        )

        b = self.b
        norm = (self.convolution.weight**2).sum((1, 2))

        for t in range(nb_steps):

            # reset term
            rst = torch.einsum("ab,b,b->ab", spk, self.b, norm)

            input_ = x[:, :, t]

            # membrane potential update
            mem = (mem - rst) * self.beta + input_ * (1.0 - self.beta)
            mthr = torch.einsum("ab,b->ab", mem, 1.0 / (norm + self.eps)) - b

            spk = self.spike_fn(mthr)

            spk_rec[:, :, t] = spk

        # save spk_rec for plotting
        #        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        if self.flatten_output:
            output = torch.transpose(spk_rec, 1, 2).contiguous()
        else:
            output = spk_rec

        return output

    def reset_parameters(self):
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1.0, std=0.01)

    def clamp(self):
        self.beta.data.clamp_(0.0, 1.0)
        self.b.data.clamp_(min=0.0)
