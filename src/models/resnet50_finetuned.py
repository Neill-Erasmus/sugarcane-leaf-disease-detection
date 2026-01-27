import torch
import torch.nn as nn
from torchvision import models

class ResNet50FineTuned(nn.Module):
    """
    Fine-tuned ResNet50 model for sugarcane leaf disease classification.

    Only the last layer of layer4 and final FC layer are trainable.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.model = models.resnet50(pretrained=True)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last block
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)