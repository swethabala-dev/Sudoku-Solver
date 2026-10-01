import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from sudoku.model import DigitCNN


# Device
device = torch.device("cpu")


# Load test dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# Load model
model = DigitCNN().to(device)

model.load_state_dict(
    torch.load(
        "sudoku/digit_cnn.pth",
        map_location=device
    )
)

model.eval()


# Store predictions and labels
all_predictions = []
all_labels = []


# Evaluate
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# Accuracy
accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")


# Classification report
print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions
    )
)


# Confusion matrix
print("\nConfusion Matrix:")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)