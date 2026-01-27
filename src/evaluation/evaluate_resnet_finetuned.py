import torch
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from src.models.resnet50_finetuned import ResNet50FineTuned
from src.data.data_loaders import get_dataloaders
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config.get("batch_size", 32)
NUM_CLASSES = config.get("num_classes", 5)
CHECKPOINT_PATH = config.get("checkpoint_path")
EXP_DIR = Path("experiments/resnet_finetuned")
EXP_DIR.mkdir(parents=True, exist_ok=True)

def evaluate() -> None:
    """
    Evaluate the fine-tuned ResNet50 model on the test set.
    Saves metrics.json and confusion_matrix.txt to the experiment folder.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet50FineTuned(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    _, _, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

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

    with open(EXP_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=4)

    with open(EXP_DIR / "confusion_matrix.txt", "w") as f:
        f.write(np.array2string(conf_matrix))

    print(f"Evaluation complete. Results saved to {EXP_DIR}/")

if __name__ == "__main__":
    evaluate()