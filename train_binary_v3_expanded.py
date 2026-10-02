import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(r"D:\e50")

DATA_DIR = BASE_DIR / "food_binary_v3_expanded"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# ============================================================
# TRAINING TRANSFORM
# Same augmentation used in Binary V3
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize(256),

    transforms.RandomResizedCrop(
        224,
        scale=(0.85, 1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# VALIDATION TRANSFORM
# IMPORTANT:
# Resize directly to 224 x 224 so every image has same size
# ============================================================
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# DATASETS
# ============================================================
train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=val_transform
)

print("\nClasses:", train_dataset.classes)

print("Training images:", len(train_dataset))

print("Validation images:", len(val_dataset))

# ============================================================
# DATALOADERS
# ============================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

# ============================================================
# MODEL
# EfficientNet-B0
# ============================================================
model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=2
)

# ============================================================
# FREEZE FEATURE EXTRACTOR
# ============================================================
for param in model.parameters():
    param.requires_grad = False

# ============================================================
# TRAIN ONLY CLASSIFIER
# ============================================================
for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)

# ============================================================
# LOSS FUNCTION
# ============================================================
criterion = nn.CrossEntropyLoss()

# ============================================================
# OPTIMIZER
# ============================================================
optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.0001
)

# ============================================================
# TRAINING SETTINGS
# ============================================================
EPOCHS = 15

best_val_accuracy = 0.0
best_epoch = 0

# IMPORTANT:
# New checkpoint so original V3 is NOT overwritten
checkpoint_path = (
    CHECKPOINT_DIR /
    "binary_food_classifier_v3_expanded.pth"
)

# ============================================================
# TRAINING LOOP
# ============================================================
for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update classifier weights
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_loss = running_loss / total

    train_accuracy = (
        100.0 * correct / total
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = (
        100.0 * val_correct / val_total
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_epoch = epoch + 1

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    train_dataset.classes,

                "best_val_accuracy":
                    best_val_accuracy,

                "epoch":
                    best_epoch
            },
            checkpoint_path
        )

        print("  --> Best model saved!")

# ============================================================
# FINAL RESULT
# ============================================================
print("\n========================================")
print("Training completed!")
print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)
print(
    f"Best Epoch: {best_epoch}"
)
print(
    f"Model saved to: "
    f"{checkpoint_path}"
)
print("========================================")