import torch
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
from src.models.baseline_cnn import BaselineCNN
from src.data.data_loaders import get_dataloaders

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(batch_size=32, img_size=256)

    model = BaselineCNN(num_classes=5).to(device)
    checkpoint_path = os.path.join("experiments", "baseline_cnn", "checkpoints", "baseline_cnn_best.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute metrics
    report = classification_report(all_labels, all_preds, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    # Save confusion matrix
    cm_path = os.path.join("experiments", "baseline_cnn", "confusion_matrix.txt")
    with open(cm_path, "w") as f:
        for row in cm:
            f.write(" ".join(str(x) for x in row) + "\n")

    # Save metrics.json
    metrics_path = os.path.join("experiments", "baseline_cnn", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)

    print("Evaluation complete. Confusion matrix and metrics saved.")

if __name__ == "__main__":
    evaluate()
