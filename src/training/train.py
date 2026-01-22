import torch
import torch.nn as nn
import torch.optim as optim
from src.models.baseline_cnn import BaselineCNN
from src.data.data_loaders import get_dataloaders
import os

#configuration
batch_size = 32
lr = 1e-3
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "experiments/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

#data
train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

#model
model = BaselineCNN(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

#training
best_val_acc = 0.0
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    #validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")
    #save checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'baseline_cnn_best.pth'))