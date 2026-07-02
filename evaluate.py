import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from src.model import KneeOAModel
from src.dataloader import test_loader


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)


# LOAD MODEL
model = KneeOAModel().to(device)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

model.eval()


all_preds = []
all_labels = []

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        all_preds.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


accuracy = 100 * correct / total

print("\n===================")
print("TEST ACCURACY:", accuracy)
print("===================\n")


print("CLASSIFICATION REPORT:\n")

print(
    classification_report(
        all_labels,
        all_preds
    )
)


print("\nCONFUSION MATRIX:\n")

print(
    confusion_matrix(
        all_labels,
        all_preds
    )
)