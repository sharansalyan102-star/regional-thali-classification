import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = "split_dataset/test"
MODEL_PATH = "checkpoints/efficientnet_augmented_original.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# TEST TRANSFORM
# NO AUGMENTATION
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

classes = test_dataset.classes

print(f"Device: {DEVICE}")
print(f"Test images: {len(test_dataset)}")
print(f"Classes: {classes}")

# ============================================================
# LOAD MODEL
# ============================================================

model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    len(classes)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print(
    f"Loaded model from epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Best validation accuracy: "
    f"{checkpoint['best_val_accuracy']:.2f}%"
)

# ============================================================
# TESTING
# ============================================================

all_predictions = []
all_labels = []

correct = 0
total = 0

print()
print("Testing...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

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

# ============================================================
# RESULTS
# ============================================================

test_accuracy = (
    100 * correct / total
)

print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)

print()
print("Classification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        digits=4
    )
)

print("Confusion Matrix:")

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print(cm)

print()
print("=" * 60)
print("Testing completed successfully!")
print("=" * 60)

print(
    f"Original EfficientNet baseline: 96.67%"
)

print(
    f"Augmented EfficientNet: {test_accuracy:.2f}%"
)

if test_accuracy > 96.67:
    print("✓ Augmentation improved test accuracy.")
elif test_accuracy < 96.67:
    print("✗ Augmentation reduced test accuracy.")
else:
    print("= Same test accuracy as baseline.")