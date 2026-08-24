import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights


# ==========================================
# SETTINGS
# ==========================================

TRAIN_DIR = "split_dataset/train"
VAL_DIR = "split_dataset/val"

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs/efficientnet"

BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.0001

IMAGE_SIZE = 224
NUM_CLASSES = 3
SEED = 42


# ==========================================
# REPRODUCIBILITY
# ==========================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# DIRECTORIES
# ==========================================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# DATA AUGMENTATION
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

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


# No random augmentation for validation

val_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
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

print("\nClasses:")
print(train_dataset.classes)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ==========================================
# DATALOADERS
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
# EFFICIENTNET-B0 MODEL
# ==========================================

weights = EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)

# Replace classifier

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)

model = model.to(device)


# ==========================================
# LOSS & OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# TRAINING
# ==========================================

best_val_accuracy = 0.0

print("\n========================================")
print("Starting EfficientNet-B0 Training")
print("========================================")


for epoch in range(NUM_EPOCHS):

    # ======================================
    # TRAIN
    # ======================================

    model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

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
        100.0 *
        train_correct /
        train_total
    )


    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

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
        100.0 *
        val_correct /
        val_total
    )


    # ======================================
    # PRINT RESULTS
    # ======================================

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: "
        f"{train_loss / len(train_loader):.4f} "
        f"Train Acc: "
        f"{train_accuracy:.2f}% "
        f"Val Loss: "
        f"{val_loss / len(val_loader):.4f} "
        f"Val Acc: "
        f"{val_accuracy:.2f}%"
    )


    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            "efficientnet_best.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_accuracy":
                    best_val_accuracy,

                "classes":
                    train_dataset.classes
            },
            checkpoint_path
        )

        print(
            f"  ✓ Best model saved "
            f"({best_val_accuracy:.2f}%)"
        )


# ==========================================
# FINISHED
# ==========================================

print("\n========================================")
print("Training Completed")
print("========================================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "Model saved at:"
    " checkpoints/efficientnet_best.pth"
)