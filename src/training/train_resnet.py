import torch
import torch.nn as nn
import torch.optim as optim
from src.models.resnet_transfer import ResNetTransfer
from src.data.data_loaders import get_dataloaders
from pathlib import Path
import yaml
import json
from tqdm import tqdm

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config["batch_size"]
LR = config["learning_rate"]
EPOCHS = 15
NUM_CLASSES = config["num_classes"]
IMG_SIZE = config["input_size"]
CHECKPOINT_PATH = Path(config["checkpoints"]["resnet_frozen"])
EXP_DIR = Path(config["experiments"]["resnet_frozen"])
EXP_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train() -> None:
    """Train ResNetTransfer on sugarcane leaf dataset."""
    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE, img_size=IMG_SIZE)

    model = ResNetTransfer(num_classes=NUM_CLASSES, freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    with open(EXP_DIR / "metrics.json", "w") as f:
        json.dump({"best_val_accuracy": best_val_acc}, f, indent=4)

if __name__ == "__main__":
    train()