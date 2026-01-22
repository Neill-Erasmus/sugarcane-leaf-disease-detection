import torch
from sklearn.metrics import confusion_matrix, classification_report
from src.models.baseline_cnn import BaselineCNN
from src.data.data_loaders import get_dataloaders
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

model = BaselineCNN(num_classes=5).to(device)
model.load_state_dict(torch.load("experiments/checkpoints/baseline_cnn_best.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes))
print("Confusion Matrix:\n")
print(confusion_matrix(all_labels, all_preds))