from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple
import yaml
import os

CONFIG_PATH = "config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {}

BATCH_SIZE = config.get("batch_size", 32)
IMG_SIZE = config.get("input_size", 256)

DATA_PATHS = {
    "train": "data/train",
    "val": "data/val",
    "test": "data/test"
}

def get_dataloaders(batch_size: int = BATCH_SIZE, img_size: int = IMG_SIZE
                   ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns PyTorch DataLoaders for training, validation, and testing datasets.

    Args:
        batch_size (int): Number of samples per batch.
        img_size (int): Size to resize images (img_size x img_size).

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train, val, test loaders
    """

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=DATA_PATHS["train"], transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=DATA_PATHS["val"], transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(root=DATA_PATHS["test"], transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader