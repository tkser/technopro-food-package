import torch
import torch.nn as nn
import torch.nn.functional as F


class PackageNet(nn.Module):
    def __init__(self):
        super(PackageNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, device='cuda:0'),
            nn.BatchNorm2d(16, device='cuda:0'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, device='cuda:0'),
            nn.BatchNorm2d(32, device='cuda:0'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(32 * 32 * 32, 32, device='cuda:0')
        self.fc2 = nn.Linear(32, 16, device='cuda:0')
        self.fc3 = nn.Linear(16, 2, device='cuda:0')

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x