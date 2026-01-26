import torch.nn as nn
from torchvision import models

class ResNet50FineTuned(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()

        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)