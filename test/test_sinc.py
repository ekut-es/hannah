import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from hannah.features import SincConv


def test_sinc():
    model = SincConv(padding=50)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    orig_parameters = {}
    for name, parameter in model.named_parameters():
        orig_parameters[name] = copy.deepcopy(parameter.detach())

    for epoch in range(20):

        input = torch.rand(16, 1, 1600)
        target = torch.rand(16, 40, 1600)
        optimizer.zero_grad()
        output = model(input)
        print(output.shape)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter, orig_parameters[name])


if __name__ == "__main__":
    test_sinc()
