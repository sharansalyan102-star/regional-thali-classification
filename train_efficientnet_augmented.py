import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ==========================================
# CONFIGURATION
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_DIR = "split_dataset/train"
VAL_DIR = "split_dataset/val"

CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.0001

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("Device:", DEVICE)

# ==========================================
# TRAIN TRANSFORM WITH AUGMENTATION
# ==========================================

train_transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        224,
        scale=(0.85, 1.0)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==========================================
# VALIDATION TRANSFORM
# NO AUGMENTATION
# ==========================================

val_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==========================================
# DATASETS
# ==========================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print()
print("Classes:")
print(train_dataset.classes)

print()
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# ==========================================
# DATA LOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==========================================
# LOAD EFFICIENTNET-B0
# ==========================================

model = models.efficientnet_b0(
    weights="DEFAULT"
)

# Freeze feature extractor initially
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Replace final classifier for 3 classes
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(train_dataset.classes)
)

model = model.to(DEVICE)

# ==========================================
# LOSS AND OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)

# ==========================================
# TRAINING
# ==========================================

best_val_accuracy = 0.0

print()
print("=" * 55)
print("Starting EfficientNet-B0 Training")
print("Data Augmentation Enabled")
print("=" * 55)

for epoch in range(EPOCHS):

    # ---------------------------
    # TRAINING
    # ---------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = (
        100 * train_correct / train_total
    )

    # ---------------------------
    # VALIDATION
    # ---------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = (
        100 * val_correct / val_total
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss / len(val_loader):.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    # ---------------------------
    # SAVE BEST MODEL
    # ---------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            "efficientnet_augmented_best.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "classes": train_dataset.classes,
                "val_accuracy": best_val_accuracy
            },
            checkpoint_path
        )

        print(
            f"  ✓ Best augmented model saved "
            f"({best_val_accuracy:.2f}%)"
        )

# ==========================================
# COMPLETED
# ==========================================

print()
print("=" * 55)
print("Training Completed")
print("=" * 55)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "Model saved at: "
    "checkpoints/efficientnet_augmented_best.pth"
)