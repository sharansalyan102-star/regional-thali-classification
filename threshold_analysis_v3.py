import os
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch.utils.data import DataLoader

# ============================================================
# PATHS
# ============================================================

FOOD_DIR = r"D:\e50\split_dataset\test"
NON_FOOD_DIR = r"D:\e50\unknown_test\non_food"
MODEL_PATH = r"D:\e50\checkpoints\binary_food_classifier_v3.pth"

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
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

food_dataset = datasets.ImageFolder(
    FOOD_DIR,
    transform=transform
)

# Non-food images are all labelled as one class
non_food_dataset = datasets.ImageFolder(
    os.path.dirname(NON_FOOD_DIR),
    transform=transform
)

# Make sure we specifically use the non_food folder
non_food_dataset = datasets.ImageFolder(
    os.path.dirname(NON_FOOD_DIR),
    transform=transform
)

print("Food images:", len(food_dataset))
print("Non-food images:", len(non_food_dataset))

# ============================================================
# MODEL
# ============================================================

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

state_dict = checkpoint.get(
    "model_state_dict",
    checkpoint
)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

food_index = classes.index("food")

# ============================================================
# GET FOOD PROBABILITIES
# ============================================================

def get_probabilities(dataset):

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False
    )

    probabilities = []

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            food_probs = probs[:, food_index]

            probabilities.extend(
                food_probs.cpu().tolist()
            )

    return probabilities


print("\nCalculating predictions...")

food_probs = get_probabilities(food_dataset)
non_food_probs = get_probabilities(non_food_dataset)

print("Predictions calculated.")

# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]

print("\n")
print("=" * 90)
print("BINARY V3 THRESHOLD ANALYSIS")
print("=" * 90)

print(
    f"{'Threshold':<12}"
    f"{'Food Accepted':<18}"
    f"{'Food Acceptance':<20}"
    f"{'Non-food Rejected':<20}"
    f"{'Non-food Rejection':<20}"
)

print("-" * 90)

for threshold in thresholds:

    # Genuine food
    food_accepted = sum(
        p >= threshold
        for p in food_probs
    )

    food_acceptance = (
        food_accepted / len(food_probs)
    ) * 100

    # Non-food
    non_food_rejected = sum(
        p < threshold
        for p in non_food_probs
    )

    non_food_rejection = (
        non_food_rejected / len(non_food_probs)
    ) * 100

    print(
        f"{threshold * 100:>6.0f}%      "
        f"{food_accepted}/{len(food_probs):<13}"
        f"{food_acceptance:>6.2f}%             "
        f"{non_food_rejected}/{len(non_food_probs):<15}"
        f"{non_food_rejection:>6.2f}%"
    )

print("=" * 90)

# ============================================================
# FALSE REJECTIONS / FALSE ACCEPTANCES
# ============================================================

print("\n")
print("=" * 70)
print("DETAILED TRADE-OFF")
print("=" * 70)

for threshold in thresholds:

    food_rejected = sum(
        p < threshold
        for p in food_probs
    )

    non_food_accepted = sum(
        p >= threshold
        for p in non_food_probs
    )

    print(
        f"\nThreshold {threshold * 100:.0f}%"
    )

    print(
        f"  Genuine food rejected : "
        f"{food_rejected}/{len(food_probs)}"
    )

    print(
        f"  Non-food accepted     : "
        f"{non_food_accepted}/{len(non_food_probs)}"
    )