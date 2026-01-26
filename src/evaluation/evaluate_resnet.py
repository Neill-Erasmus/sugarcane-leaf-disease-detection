import json
import torch
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
from src.models.resnet_transfer import ResNetTransfer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENT_DIR = Path("experiments/resnet_frozen")
CHECKPOINT_PATH = Path("experiments/checkpoints/resnet_frozen_best.pth")

def evaluate():
    tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_ds = ImageFolder("data/test", transform=tfms)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = ResNetTransfer(num_classes=5, freeze_backbone=True).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    report = classification_report(y_true, y_pred, target_names=test_ds.classes, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)

    with open(EXPERIMENT_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=4)

    with open(EXPERIMENT_DIR / "confusion_matrix.txt", "w") as f:
        f.write(str(matrix))

    print(classification_report(y_true, y_pred, target_names=test_ds.classes))
    print("\nConfusion Matrix:\n", matrix)

if __name__ == "__main__":
    evaluate()