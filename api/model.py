import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

CLASS_NAMES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def load_model():
    model = ResNet50FineTuned(num_classes=5)

    checkpoint_path = (
        "experiments/resnet_finetuned/checkpoints/resnet50_finetuned_best.pth"
    )
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

def preprocess(img_bytes):
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    tensor = transform(image).unsqueeze(0)
    return tensor.to(DEVICE)

def predict(model, img_bytes):
    x = preprocess(img_bytes)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    return {
        "predicted_class": CLASS_NAMES[int(probs.argmax())],
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
    }