import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """
    Simple convolutional neural network for sugarcane leaf disease classification.

    Architecture:
        Conv(3->32) -> ReLU -> MaxPool
        Conv(32->64) -> ReLU -> MaxPool
        Conv(64->128) -> ReLU -> MaxPool
        Fully connected: 128*32*32 -> 256 -> num_classes
        Dropout applied after conv and FC layers
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x