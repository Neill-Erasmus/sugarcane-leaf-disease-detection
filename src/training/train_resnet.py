import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.models.resnet_transfer import ResNetTransfer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 15
BATCH_SIZE = 32
LR = 1e-3

EXPERIMENT_DIR = Path("experiments/resnet_frozen")
CHECKPOINT_PATH = Path("experiments/checkpoints/resnet_frozen_best.pth")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

def get_dataloaders():
    train_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_ds = ImageFolder("data/train", transform=train_tfms)
    val_ds = ImageFolder("data/val", transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    return train_loader, val_loader

def train():
    train_loader, val_loader = get_dataloaders()

    model = ResNetTransfer(num_classes=5, freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    with open(EXPERIMENT_DIR / "metrics.json", "w") as f:
        json.dump({"best_val_accuracy": best_acc}, f, indent=4)

if __name__ == "__main__":
    train()