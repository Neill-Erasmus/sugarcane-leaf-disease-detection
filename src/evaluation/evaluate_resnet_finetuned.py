import torch
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from src.models.resnet50_finetuned import ResNet50FineTuned
from src.data.data_loaders import get_dataloaders

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet50FineTuned(num_classes=5)
    model.to(device)

    checkpoint_path = (
        "experiments/resnet_finetuned/checkpoints/"
        "resnet50_finetuned_best.pth"
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    _, _, test_loader = get_dataloaders(batch_size=32)

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    exp_dir = Path("experiments/resnet_finetuned")

    with open(exp_dir / "confusion_matrix.txt", "w") as f:
        f.write(np.array2string(conf_matrix))

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Evaluation complete.")
    print("Results saved to experiments/resnet_finetuned/")

if __name__ == "__main__":
    evaluate()