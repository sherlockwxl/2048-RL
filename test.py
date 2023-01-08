import torch
from torch import nn
import numpy as np
from torchsummary import summary

input = torch.randn(1, 1, 4, 4)
# With default parameters
c = 1
output_dim = 4
m = nn.Sequential(
    nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.Linear(64, 16),
    nn.ReLU(),
    nn.Linear(16, output_dim),

)
print(summary(m, (1,1, 4, 4)))
output = m(input)
print(output.size())

