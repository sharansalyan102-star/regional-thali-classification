import os
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch.utils.data import DataLoader

# ============================================================
# PATHS
# ============================================================

TEST_DIR = r"D:\e50\split_dataset\test"
MODEL_PATH = r"D:\e50\checkpoints\binary_food_classifier_v3.pth"

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# TRANSFORM
# Same clean transform used for testing
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
# DATASET
# ============================================================

dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False
)

print("Thali classes:", dataset.classes)
print("Total food images:", len(dataset))

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Binary V3...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

classes = checkpoint.get(
    "classes",
    ["food", "non_food"]
)

print("Binary classes:", classes)

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

state_dict = checkpoint.get("model_state_dict", checkpoint)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("Model loaded successfully.")

# ============================================================
# TEST
# ============================================================

correct_food = 0
incorrect_rejected = 0

# Per thali class
class_stats = {
    class_name: {
        "total": 0,
        "accepted": 0,
        "rejected": 0
    }
    for class_name in dataset.classes
}

food_index = classes.index("food")

with torch.no_grad():

    image_index = 0

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        predictions = torch.argmax(probabilities, dim=1)

        for i in range(len(images)):

            pred = predictions[i].item()

            actual_thali_index = labels[i].item()
            actual_thali = dataset.classes[actual_thali_index]

            class_stats[actual_thali]["total"] += 1

            if pred == food_index:
                correct_food += 1
                class_stats[actual_thali]["accepted"] += 1
            else:
                incorrect_rejected += 1
                class_stats[actual_thali]["rejected"] += 1

            image_index += 1

# ============================================================
# RESULTS
# ============================================================

total = len(dataset)

food_acceptance_rate = (
    correct_food / total
) * 100

food_false_rejection_rate = (
    incorrect_rejected / total
) * 100

print("\n" + "=" * 60)
print("BINARY V3 — FOOD ACCEPTANCE TEST")
print("=" * 60)

print(f"Total genuine food images : {total}")
print(f"Correctly accepted as food: {correct_food}")
print(f"Incorrectly rejected      : {incorrect_rejected}")

print(f"\nFood Acceptance Rate      : {food_acceptance_rate:.2f}%")
print(f"Food False Rejection Rate : {food_false_rejection_rate:.2f}%")

print("\n" + "-" * 60)
print("PER-THALI RESULTS")
print("-" * 60)

for class_name, stats in class_stats.items():

    acceptance = (
        stats["accepted"] / stats["total"]
    ) * 100

    print(f"\n{class_name}")
    print(f"  Total    : {stats['total']}")
    print(f"  Accepted : {stats['accepted']}")
    print(f"  Rejected : {stats['rejected']}")
    print(f"  Acceptance Rate: {acceptance:.2f}%")

print("\n" + "=" * 60)
print("FINAL INTERPRETATION")
print("=" * 60)

print(
    f"Binary V3 accepts {food_acceptance_rate:.2f}% "
    "of genuine thali images as FOOD."
)

print(
    f"It incorrectly rejects {food_false_rejection_rate:.2f}% "
    "of genuine thali images."
)