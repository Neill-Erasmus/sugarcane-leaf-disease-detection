import torch
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.models.resnet_transfer import ResNetTransfer
from src.data.data_loaders import get_dataloaders

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(batch_size=32)

    model = ResNetTransfer(num_classes=5)
    model.to(device)

    checkpoint_path = "experiments/resnet_frozen/checkpoints/resnet_frozen_best.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    # Save metrics
    with open("experiments/resnet_frozen/metrics.json", "w") as f:
        json.dump(report, f, indent=4)

    # Save confusion matrix
    np.savetxt(
        "experiments/resnet_frozen/confusion_matrix.txt",
        cm,
        fmt="%d"
    )

    print("Evaluation complete — results saved to experiments/resnet_frozen/")

if __name__ == "__main__":
    evaluate()
