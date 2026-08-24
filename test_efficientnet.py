import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# SETTINGS
# ==========================================

TEST_DIR = "split_dataset/test"
CHECKPOINT_PATH = "checkpoints/efficientnet_best.pth"
OUTPUT_DIR = "outputs/efficientnet"

BATCH_SIZE = 16
IMAGE_SIZE = 224

# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# TEST TRANSFORM
# ==========================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# TEST DATASET
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

classes = test_dataset.classes

print("Test images:", len(test_dataset))
print("Classes:", classes)

# ==========================================
# LOAD EFFICIENTNET-B0
# ==========================================

model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print(
    "Loaded model from epoch:",
    checkpoint["epoch"]
)

print(
    "Best validation accuracy:",
    f"{checkpoint['val_accuracy']:.2f}%"
)

# ==========================================
# TESTING
# ==========================================

all_predictions = []
all_labels = []

correct = 0
total = 0

print("\nTesting...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
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

# ==========================================
# ACCURACY
# ==========================================

test_accuracy = (
    100.0 * correct / total
)

print("\nTest Accuracy:")
print(f"{test_accuracy:.2f}%")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=classes,
    digits=4
)

print("\nClassification Report:")
print(report)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)

# ==========================================
# SAVE CLASSIFICATION REPORT
# ==========================================

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("EfficientNet-B0 Classification Report\n")
    f.write("=" * 50 + "\n\n")

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy:.2f}%\n\n"
    )

    f.write(report)

    f.write("\n\nConfusion Matrix:\n")

    f.write(
        np.array2string(cm)
    )

# ==========================================
# SAVE TEST RESULTS
# ==========================================

results_path = os.path.join(
    OUTPUT_DIR,
    "test_results.txt"
)

with open(
    results_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy:.2f}%\n\n"
    )

    f.write("Classification Report:\n")
    f.write(report)

    f.write("\n\nConfusion Matrix:\n")
    f.write(np.array2string(cm))

# ==========================================
# CONFUSION MATRIX IMAGE
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45
)

plt.title(
    "EfficientNet-B0 Confusion Matrix"
)

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# FINAL
# ==========================================

print("\n========================================")
print("Testing completed successfully!")
print("========================================")

print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)

print(
    f"Classification report saved to:"
    f"\n{report_path}"
)

print(
    f"\nConfusion matrix saved to:"
    f"\n{cm_path}"
)

print(
    f"\nTest results saved to:"
    f"\n{results_path}"
)