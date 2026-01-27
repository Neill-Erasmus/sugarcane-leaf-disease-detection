import torch
import torch.nn as nn
from torchvision import models

class ResNetTransfer(nn.Module):
    """
    ResNet50 model with transfer learning.

    Args:
        num_classes (int): Number of output classes.
        freeze_backbone (bool): If True, freeze all pretrained layers except the classifier.
    """

    def __init__(self, num_classes: int = 5, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features

        # New classifier
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)