import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import yaml
import os
 
CONFIG_PATH = "config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    CHECKPOINT_PATH = config.get(
        "checkpoint_path", "experiments/resnet_finetuned/checkpoints/resnet50_finetuned_best.pth"
    )
else:
    CHECKPOINT_PATH = "experiments/resnet_finetuned/checkpoints/resnet50_finetuned_best.pth"

CLASS_NAMES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNet50FineTuned(nn.Module):
    """
    ResNet50 model fine-tuned for sugarcane leaf disease classification.
    Freezes all layers except layer4 and replaces the final classifier.
    """

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

def load_model() -> nn.Module:
    """
    Load the fine-tuned ResNet50 model and set it to evaluation mode.

    Returns:
        nn.Module: Loaded PyTorch model.
    """

    model = ResNet50FineTuned(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def preprocess(img_bytes: bytes) -> torch.Tensor:
    """
    Preprocess an image byte stream for model prediction.

    Args:
        img_bytes (bytes): Image in bytes.

    Returns:
        torch.Tensor: Preprocessed tensor ready for model input.
    """
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    tensor = transform(image).unsqueeze(0)
    return tensor.to(DEVICE)

def predict(model: nn.Module, img_bytes: bytes) -> dict:
    """
    Predict class and probabilities from an image using the given model.

    Args:
        model (nn.Module): Loaded PyTorch model.
        img_bytes (bytes): Image in bytes.

    Returns:
        dict: Dictionary containing predicted class and probabilities.
    """
    
    x = preprocess(img_bytes)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    return {
        "predicted_class": CLASS_NAMES[int(probs.argmax())],
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }