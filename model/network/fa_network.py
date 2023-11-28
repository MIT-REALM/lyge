import torch
import torch.nn as nn


class FaNetwork(nn.Module):
    """Ground effect network"""

    def __init__(self):
        super(FaNetwork, self).__init__()
        self.fc1 = nn.Linear(12, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, 3)

    def forward(self, x):
        if not x.is_cuda:
            self.cpu()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def disable_grad(self):
        for i in self.parameters():
            i.requires_grad = False
