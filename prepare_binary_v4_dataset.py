import os
import random
import shutil
from pathlib import Path

random.seed(42)

BASE = Path(r"D:\e50")

# Existing food dataset
OLD_DATASET = BASE / "food_binary_split"

# New 70 non-food images
NEW_NONFOOD = BASE / "nonfood"

# New V4 dataset
OUT = BASE / "food_binary_v4"

# Create folders
for split in ["train", "val", "test"]:
    for cls in ["food", "non_food"]:
        (OUT / split / cls).mkdir(parents=True, exist_ok=True)

# -------------------------
# Copy FOOD (existing split)
# -------------------------
for split in ["train", "val", "test"]:
    src = OLD_DATASET / split / "food"
    dst = OUT / split / "food"

    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)

print("Food images copied.")

# -------------------------
# Split NEW non-food
# -------------------------
images = [
    f for f in NEW_NONFOOD.iterdir()
    if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

random.shuffle(images)

train_imgs = images[:55]
val_imgs = images[55:]

for img in train_imgs:
    shutil.copy2(img, OUT / "train" / "non_food" / img.name)

for img in val_imgs:
    shutil.copy2(img, OUT / "val" / "non_food" / img.name)

print("Non-food split completed.")
print(f"Train: {len(train_imgs)}")
print(f"Val  : {len(val_imgs)}")

print("\nV4 dataset created successfully!")