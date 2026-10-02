import os
import torch
import timm
from torchvision import transforms
from PIL import Image

CHECKPOINT = r"D:\e50\checkpoints\binary_food_classifier_v3_expanded.pth"

# Change this only if your 90-image test folder has a different path
FOOD_DIR = r"D:\e50\split_dataset\test"

device = torch.device("cpu")

checkpoint = torch.load(CHECKPOINT, map_location=device)
classes = checkpoint["classes"]

print("Classes:", classes)

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

food_index = classes.index("food")

total = 0
accepted = 0
rejected = 0

print("\n========================================")
print("Testing V3 Expanded on Genuine Thali Images")
print("========================================")

for root, dirs, files in os.walk(FOOD_DIR):

    for filename in sorted(files):

        filepath = os.path.join(root, filename)

        try:
            image = Image.open(filepath).convert("RGB")
        except Exception:
            continue

        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            food_probability = probabilities[0][food_index].item() * 100

        total += 1

        if food_probability >= 50:
            accepted += 1
            result = "ACCEPTED"
        else:
            rejected += 1
            result = "REJECTED"

        print(
            f"{filename:40} "
            f"Food: {food_probability:6.2f}% -> {result}"
        )

print("\n========================================")
print("RESULT")
print("========================================")
print(f"Total genuine thali images : {total}")
print(f"Accepted as food           : {accepted}")
print(f"Rejected as non-food       : {rejected}")
print(f"Food acceptance rate       : {accepted / total * 100:.2f}%")
print(f"False rejection rate       : {rejected / total * 100:.2f}%")
print("========================================")