import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "checkpoints/binary_food_classifier_v3.pth"
NON_FOOD_DIR = "unknown_test/non_food"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = ["food", "non_food"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print(f"Device: {DEVICE}")
print("Loading Binary V3...")

model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    2
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

print("Model loaded successfully.")
print(f"Classes: {CLASSES}")
print(f"Best validation accuracy: "
      f"{checkpoint['best_val_accuracy']:.2f}%")

image_files = [
    f for f in os.listdir(NON_FOOD_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    )
]

print()
print("=" * 70)
print("TESTING BINARY V3 ON NON-FOOD IMAGES")
print("=" * 70)

print(f"Total images: {len(image_files)}")
print()

correctly_rejected = 0
incorrectly_accepted = 0

with torch.no_grad():

    for i, filename in enumerate(sorted(image_files), 1):

        path = os.path.join(
            NON_FOOD_DIR,
            filename
        )

        image = Image.open(path).convert("RGB")

        image_tensor = transform(
            image
        ).unsqueeze(0).to(DEVICE)

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

        predicted_class = CLASSES[
            predicted.item()
        ]

        confidence = confidence.item() * 100

        if predicted_class == "non_food":

            correctly_rejected += 1

            status = "✓ CORRECTLY REJECTED"

        else:

            incorrectly_accepted += 1

            status = "✗ INCORRECTLY ACCEPTED"

        print(
            f"{i:02d}. "
            f"{filename[:30]:30} | "
            f"{predicted_class:10} | "
            f"{confidence:6.2f}% | "
            f"{status}"
        )

total = len(image_files)

rejection_rate = (
    correctly_rejected / total * 100
)

print()
print("=" * 70)
print("FINAL BINARY V3 RESULTS")
print("=" * 70)

print(
    f"Total non-food images       : {total}"
)

print(
    f"Correctly rejected          : "
    f"{correctly_rejected}"
)

print(
    f"Incorrectly accepted        : "
    f"{incorrectly_accepted}"
)

print(
    f"Non-food rejection rate     : "
    f"{rejection_rate:.2f}%"
)

print("=" * 70)