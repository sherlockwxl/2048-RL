from torch import nn
import copy
from torchvision import models
from torchsummary import summary


class CustNet(nn.Module):
    '''mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    '''

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 4:
            raise ValueError(f"Expecting input height: 4, got: {h}")
        if w != 4:
            raise ValueError(f"Expecting input width: 4, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )
        #print(summary(self.online, (1, 4, 4)))
        # exit()

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == 'online':
            #print(input.size())
            return self.online(input)
        elif model == 'target':
            return self.target(input)
