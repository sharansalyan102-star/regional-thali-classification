import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =========================
# Configuration
# =========================
MODEL_PATH = "checkpoints/efficientnet_best.pth"
NON_FOOD_DIR = "unknown_test/non_food"

CLASSES = [
    "Bengali thali",
    "Kerala Sadhya",
    "Rajasthani"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Image transformation
# Same preprocessing as original EfficientNet
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Load EfficientNet-B0
# =========================
print(f"Device: {DEVICE}")

model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(CLASSES))

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print(f"Classes: {CLASSES}")
print()

# =========================
# Test non-food images
# =========================
image_files = [
    f for f in os.listdir(NON_FOOD_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

print(f"Non-food test images: {len(image_files)}")
print()
print("=" * 75)
print(f"{'Image':35} {'Prediction':20} {'Confidence':>12}")
print("=" * 75)

accepted = 0
confidence_sum = 0

results = []

with torch.no_grad():

    for filename in sorted(image_files):

        image_path = os.path.join(NON_FOOD_DIR, filename)

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)

            output = model(image_tensor)

            probabilities = torch.softmax(output, dim=1)

            confidence, predicted = torch.max(probabilities, dim=1)

            confidence = confidence.item() * 100
            predicted_class = CLASSES[predicted.item()]

            accepted += 1
            confidence_sum += confidence

            results.append(
                (filename, predicted_class, confidence)
            )

            print(
                f"{filename[:35]:35} "
                f"{predicted_class:20} "
                f"{confidence:10.2f}%"
            )

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# =========================
# Summary
# =========================
print("=" * 75)

total = len(results)

if total > 0:
    average_confidence = confidence_sum / total

    print()
    print("SUMMARY")
    print("=" * 40)
    print(f"Total non-food images: {total}")
    print(f"Images classified as a thali: {total}")
    print(f"Average confidence: {average_confidence:.2f}%")

print()
print("IMPORTANT:")
print("This test shows how the 3-class model behaves on non-food images.")
print("The model currently has no explicit 'Non-Food' class.")