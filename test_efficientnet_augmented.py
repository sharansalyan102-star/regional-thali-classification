import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# CONFIGURATION
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_DIR = "split_dataset/test"

MODEL_PATH = (
    "checkpoints/efficientnet_augmented_best.pth"
)

BATCH_SIZE = 8

print("Device:", DEVICE)

# ==========================================
# TEST TRANSFORM
# NO AUGMENTATION
# ==========================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==========================================
# LOAD TEST DATASET
# ==========================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))
print("Classes:", test_dataset.classes)

# ==========================================
# LOAD MODEL
# ==========================================

model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(test_dataset.classes)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print(
    "Loaded model from epoch:",
    checkpoint["epoch"]
)

print(
    "Best validation accuracy:",
    f'{checkpoint["val_accuracy"]:.2f}%'
)

# ==========================================
# TESTING
# ==========================================

all_predictions = []
all_labels = []

print()
print("Testing...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

# ==========================================
# ACCURACY
# ==========================================

correct = sum(
    p == l
    for p, l in zip(
        all_predictions,
        all_labels
    )
)

accuracy = (
    correct / len(all_labels)
) * 100

print()
print("=" * 45)
print("TEST RESULTS")
print("=" * 45)

print(
    f"Test Accuracy: {accuracy:.2f}%"
)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print()
print("Classification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        digits=4
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

print("Confusion Matrix:")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)

print()
print("=" * 45)
print("Testing completed successfully!")
print("=" * 45)