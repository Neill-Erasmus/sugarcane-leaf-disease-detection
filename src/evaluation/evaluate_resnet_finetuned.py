import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.models.resnet50_finetuned import ResNet50FineTuned
from src.data.data_loaders import get_dataloaders

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(
        img_size=224,
        batch_size=32
    )

    model = ResNet50FineTuned(num_classes=5).to(device)
    model.load_state_dict(
        torch.load(
            "experiments/checkpoints/resnet50_finetuned_best.pth",
            map_location=device
        )
    )

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:\n")
    print(classification_report(all_labels, all_preds))

    print("Confusion Matrix:\n")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()