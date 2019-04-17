from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)


x = torch.rand(5, 3)
print(x)



y = torch.rand(5, 3)
print(x + y)

device = torch.device("cuda")          # a CUDA device object
y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
x = x.to(device)                       # or just use strings ``.to("cuda")``
z = x + y
print(z)
print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
