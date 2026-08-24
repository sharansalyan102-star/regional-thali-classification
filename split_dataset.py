import os
import shutil
import random

# ==========================================
# SETTINGS
# ==========================================

SOURCE_DIR = "Dataset1"
OUTPUT_DIR = "split_dataset"

CLASSES = [
    "Bengali thali",
    "Kerala Sadhya",
    "Rajasthani"
]

# Exact number of images
TRAIN_COUNT = 144
VAL_COUNT = 31
TEST_COUNT = 30

# Fixed seed so the same split is reproduced
SEED = 42

random.seed(SEED)


# ==========================================
# SPLIT ONE CLASS
# ==========================================

def split_class(class_name):

    source_folder = os.path.join(
        SOURCE_DIR,
        class_name
    )

    # Check source folder
    if not os.path.exists(source_folder):
        print(f"ERROR: Folder not found: {source_folder}")
        return False

    # Get image files
    images = [
        file
        for file in os.listdir(source_folder)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    # Check image count
    total = len(images)

    print(f"\nChecking {class_name}...")
    print(f"Images found: {total}")

    if total < TRAIN_COUNT + VAL_COUNT + TEST_COUNT:
        print(
            f"ERROR: {class_name} needs at least "
            f"{TRAIN_COUNT + VAL_COUNT + TEST_COUNT} images."
        )
        return False

    if total > TRAIN_COUNT + VAL_COUNT + TEST_COUNT:
        print(
            f"WARNING: {class_name} has {total} images."
        )
        print(
            f"Only the first {TRAIN_COUNT + VAL_COUNT + TEST_COUNT} "
            f"after shuffling will be used."
        )

    # Shuffle images
    random.shuffle(images)

    # Use exactly 205 images
    images = images[:TRAIN_COUNT + VAL_COUNT + TEST_COUNT]

    # Split
    train_images = images[:TRAIN_COUNT]

    val_images = images[
        TRAIN_COUNT:
        TRAIN_COUNT + VAL_COUNT
    ]

    test_images = images[
        TRAIN_COUNT + VAL_COUNT:
    ]

    # ======================================
    # CREATE DESTINATION FOLDERS
    # ======================================

    train_folder = os.path.join(
        OUTPUT_DIR,
        "train",
        class_name
    )

    val_folder = os.path.join(
        OUTPUT_DIR,
        "val",
        class_name
    )

    test_folder = os.path.join(
        OUTPUT_DIR,
        "test",
        class_name
    )

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # ======================================
    # COPY TRAIN IMAGES
    # ======================================

    for image in train_images:

        source = os.path.join(
            source_folder,
            image
        )

        destination = os.path.join(
            train_folder,
            image
        )

        shutil.copy2(source, destination)

    # ======================================
    # COPY VALIDATION IMAGES
    # ======================================

    for image in val_images:

        source = os.path.join(
            source_folder,
            image
        )

        destination = os.path.join(
            val_folder,
            image
        )

        shutil.copy2(source, destination)

    # ======================================
    # COPY TEST IMAGES
    # ======================================

    for image in test_images:

        source = os.path.join(
            source_folder,
            image
        )

        destination = os.path.join(
            test_folder,
            image
        )

        shutil.copy2(source, destination)

    # ======================================
    # DISPLAY RESULTS
    # ======================================

    print(f"\n{class_name}")
    print("--------------------------------")
    print(f"Total      : {len(images)}")
    print(f"Train      : {len(train_images)}")
    print(f"Validation : {len(val_images)}")
    print(f"Test       : {len(test_images)}")

    return True


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("========================================")
    print("       THALI DATASET SPLITTING")
    print("========================================")

    print(f"Source folder : {SOURCE_DIR}")
    print(f"Output folder : {OUTPUT_DIR}")
    print(f"Random seed   : {SEED}")

    print("\nRequired split per class:")
    print("Train      : 144")
    print("Validation : 31")
    print("Test       : 30")
    print("Total      : 205")

    # Check source directory
    if not os.path.exists(SOURCE_DIR):

        print(
            f"\nERROR: Source directory "
            f"'{SOURCE_DIR}' not found."
        )

        return

    # ======================================
    # PROCESS ALL CLASSES
    # ======================================

    success = True

    for class_name in CLASSES:

        result = split_class(class_name)

        if not result:
            success = False

    # ======================================
    # FINAL MESSAGE
    # ======================================

    print("\n========================================")

    if success:

        print("Dataset split completed successfully!")

        print("\nCreated structure:")
        print("split_dataset/")
        print("├── train/")
        print("├── val/")
        print("└── test/")

        print("\nEach class:")
        print("144 Train")
        print("31 Validation")
        print("30 Test")

        print("\nTotal images:")
        print("Train      : 432")
        print("Validation : 93")
        print("Test       : 90")
        print("Total      : 615")

    else:

        print("Dataset split completed with errors.")

    print("========================================")


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    main()