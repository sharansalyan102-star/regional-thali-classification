import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================================
# SETTINGS & PATHS
# ============================================================
EFF_MODEL_PATH = r"D:\e50\checkpoints\efficientnet_best.pth"
BINARY_MODEL_PATH = r"D:\e50\checkpoints\binary_food_classifier_v3.pth"

GENUINE_THALI_DIR = r"D:\e50\split_dataset\test"
NON_FOOD_DIR = r"D:\e50\unknown_test\non_food"

BINARY_THRESHOLD = 0.50  # 50% threshold for food acceptance

# Classes
THALI_CLASSES = ["Bengali thali", "Kerala Sadhya", "Rajasthani"]
BINARY_CLASSES = ["food", "non_food"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================
# TRANSFORMS (Exact standard PyTorch / EfficientNet transform)
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
# LOAD MODELS
# ============================================================
print("\nLoading models...")

# 1. Binary V3 Classifier
binary_model = models.efficientnet_b0(weights=None)
binary_model.classifier[1] = nn.Linear(binary_model.classifier[1].in_features, len(BINARY_CLASSES))

binary_ckpt = torch.load(BINARY_MODEL_PATH, map_location=DEVICE)
binary_state_dict = binary_ckpt.get("model_state_dict", binary_ckpt)
binary_model.load_state_dict(binary_state_dict)
binary_model = binary_model.to(DEVICE)
binary_model.eval()
print(f"Loaded Binary V3 model from '{BINARY_MODEL_PATH}'")

# 2. Original 3-class EfficientNet-B0 Model
eff_model = models.efficientnet_b0(weights=None)
eff_model.classifier[1] = nn.Linear(eff_model.classifier[1].in_features, len(THALI_CLASSES))

eff_ckpt = torch.load(EFF_MODEL_PATH, map_location=DEVICE)
eff_state_dict = eff_ckpt.get("model_state_dict", eff_ckpt)
eff_model.load_state_dict(eff_state_dict)
eff_model = eff_model.to(DEVICE)
eff_model.eval()
print(f"Loaded EfficientNet-B0 model from '{EFF_MODEL_PATH}'")

# Index of 'food' in Binary V3
food_idx = BINARY_CLASSES.index("food")

# Helper function to predict with Binary V3
def predict_binary(image_tensor):
    with torch.no_grad():
        output = binary_model(image_tensor)
        probs = torch.softmax(output, dim=1)
        food_prob = probs[0, food_idx].item()
    return food_prob

# Helper function to predict with EfficientNet-B0
def predict_efficientnet(image_tensor):
    with torch.no_grad():
        output = eff_model(image_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        pred_class = THALI_CLASSES[pred_idx.item()]
        confidence = conf.item() * 100.0
    return pred_class, confidence, pred_idx.item()

# ============================================================
# 1. TEST GENUINE THALI TEST SET (90 Images)
# ============================================================
print("\n" + "=" * 90)
print("PROCESSING GENUINE THALI TEST SET")
print("=" * 90)

thali_results = []
per_class_thali_stats = {
    c: {
        "total": 0,
        "accepted": 0,
        "rejected": 0,
        "eff_correct_among_accepted": 0,
        "eff_correct_alone": 0,
        "overall_correct": 0
    }
    for c in THALI_CLASSES
}

for actual_class in THALI_CLASSES:
    class_folder = os.path.join(GENUINE_THALI_DIR, actual_class)
    if not os.path.exists(class_folder):
        print(f"Warning: Directory not found: {class_folder}")
        continue
    
    files = sorted([f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    
    for filename in files:
        img_path = os.path.join(class_folder, filename)
        image = Image.open(img_path).convert("RGB")
        img_tensor = test_transform(image).unsqueeze(0).to(DEVICE)
        
        # Binary prediction
        food_prob = predict_binary(img_tensor)
        binary_decision = "FOOD" if food_prob >= BINARY_THRESHOLD else "NON-FOOD"
        
        # EfficientNet prediction alone (for baseline comparison)
        eff_alone_class, eff_alone_conf, _ = predict_efficientnet(img_tensor)
        is_eff_alone_correct = (eff_alone_class == actual_class)
        
        # Combined pipeline
        if binary_decision == "FOOD":
            eff_class, eff_conf, _ = predict_efficientnet(img_tensor)
            is_eff_correct = (eff_class == actual_class)
            overall_correct = is_eff_correct
        else:
            eff_class = "REJECTED (Non-food)"
            eff_conf = 0.0
            is_eff_correct = False
            overall_correct = False
            
        # Update per-class stats
        stats = per_class_thali_stats[actual_class]
        stats["total"] += 1
        if is_eff_alone_correct:
            stats["eff_correct_alone"] += 1
            
        if binary_decision == "FOOD":
            stats["accepted"] += 1
            if is_eff_correct:
                stats["eff_correct_among_accepted"] += 1
                stats["overall_correct"] += 1
        else:
            stats["rejected"] += 1
            
        thali_results.append({
            "filename": filename,
            "actual_class": actual_class,
            "food_prob": food_prob,
            "binary_decision": binary_decision,
            "eff_alone_class": eff_alone_class,
            "eff_alone_conf": eff_alone_conf,
            "eff_class": eff_class,
            "eff_conf": eff_conf,
            "is_eff_correct": is_eff_correct,
            "overall_correct": overall_correct
        })
        
        print(f"[{actual_class:15}] {filename[:30]:30} | Binary: {food_prob*100:6.2f}% ({binary_decision:8}) | EffNet: {eff_class:15} ({eff_conf:6.2f}%) | Pipeline Result: {'PASS' if overall_correct else 'FAIL'}")

# ============================================================
# 2. TEST NON-FOOD TEST SET (37 Images)
# ============================================================
print("\n" + "=" * 90)
print("PROCESSING NON-FOOD TEST SET")
print("=" * 90)

non_food_results = []
non_food_files = sorted([f for f in os.listdir(NON_FOOD_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])

for filename in non_food_files:
    img_path = os.path.join(NON_FOOD_DIR, filename)
    image = Image.open(img_path).convert("RGB")
    img_tensor = test_transform(image).unsqueeze(0).to(DEVICE)
    
    food_prob = predict_binary(img_tensor)
    binary_decision = "FOOD" if food_prob >= BINARY_THRESHOLD else "NON-FOOD"
    
    if binary_decision == "FOOD":
        eff_class, eff_conf, _ = predict_efficientnet(img_tensor)
    else:
        eff_class = "N/A (Rejected)"
        eff_conf = 0.0
        
    non_food_results.append({
        "filename": filename,
        "food_prob": food_prob,
        "binary_decision": binary_decision,
        "eff_class": eff_class,
        "eff_conf": eff_conf
    })
    
    status_str = "REJECTED (Correct)" if binary_decision == "NON-FOOD" else f"INCORRECTLY ACCEPTED -> EffNet: {eff_class} ({eff_conf:.2f}%)"
    print(f"[Non-food] {filename[:30]:30} | Food Prob: {food_prob*100:6.2f}% | Decision: {binary_decision:8} | Status: {status_str}")


# ============================================================
# COMPUTE STATISTICAL SUMMARIES
# ============================================================

# Section A Stats
total_thali = len(thali_results)
thali_accepted = sum(1 for r in thali_results if r["binary_decision"] == "FOOD")
thali_rejected = total_thali - thali_accepted
thali_acceptance_rate = (thali_accepted / total_thali * 100.0) if total_thali > 0 else 0.0

accepted_thali_correct = sum(1 for r in thali_results if r["binary_decision"] == "FOOD" and r["is_eff_correct"])
eff_acc_among_accepted = (accepted_thali_correct / thali_accepted * 100.0) if thali_accepted > 0 else 0.0
overall_thali_correct = sum(1 for r in thali_results if r["overall_correct"])
overall_thali_accuracy = (overall_thali_correct / total_thali * 100.0) if total_thali > 0 else 0.0

eff_alone_correct = sum(1 for r in thali_results if r["eff_alone_class"] == r["actual_class"])
eff_alone_accuracy = (eff_alone_correct / total_thali * 100.0) if total_thali > 0 else 0.0

# Section B Stats
total_non_food = len(non_food_results)
non_food_rejected = sum(1 for r in non_food_results if r["binary_decision"] == "NON-FOOD")
non_food_accepted = total_non_food - non_food_rejected
non_food_rejection_rate = (non_food_rejected / total_non_food * 100.0) if total_non_food > 0 else 0.0

# ============================================================
# PRINT FINAL REPORT
# ============================================================

print("\n" + "=" * 90)
print("FINAL TEST REPORT: TWO-STAGE COMBINED PIPELINE (Threshold = 50%)")
print("=" * 90)

print("\nA. GENUINE FOOD / THALI IMAGES (split_dataset/test):")
print("-" * 70)
print(f"Total Genuine Thali Images                      : {total_thali}")
print(f"Accepted by Binary V3 (Food Prob >= 50%)        : {thali_accepted}")
print(f"Rejected by Binary V3 (Food Prob < 50%)         : {thali_rejected}")
print(f"Food Acceptance Rate                            : {thali_acceptance_rate:.2f}%")
print(f"Accepted Images Correctly Classified by EffNet  : {accepted_thali_correct} / {thali_accepted}")
print(f"EfficientNet Accuracy Among Accepted Images     : {eff_acc_among_accepted:.2f}%")
print(f"Overall Correct Final Predictions               : {overall_thali_correct} / {total_thali} ({overall_thali_accuracy:.2f}%)")

print("\nPer-Class Breakdown for Genuine Thali Images:")
print(f"{'Class Name':<18} | {'Total':<6} | {'Accepted':<8} | {'Rejected':<8} | {'EffNet Correct':<15} | {'Final Correct':<14} | {'Accuracy':<8}")
print("-" * 88)
for c in THALI_CLASSES:
    st = per_class_thali_stats[c]
    cls_acc = (st["overall_correct"] / st["total"] * 100.0) if st["total"] > 0 else 0.0
    print(f"{c:<18} | {st['total']:<6} | {st['accepted']:<8} | {st['rejected']:<8} | {st['eff_correct_among_accepted']:<15} | {st['overall_correct']:<14} | {cls_acc:6.2f}%")

print("\n" + "-" * 70)
print("B. NON-FOOD IMAGES (unknown_test/non_food):")
print("-" * 70)
print(f"Total Non-Food Images                           : {total_non_food}")
print(f"Correctly Rejected by Binary V3 (Food < 50%)    : {non_food_rejected}")
print(f"Incorrectly Accepted by Binary V3 (Food >= 50%) : {non_food_accepted}")
print(f"Non-Food Rejection Rate                         : {non_food_rejection_rate:.2f}%")

if non_food_accepted > 0:
    print("\nDetails for Incorrectly Accepted Non-Food Images:")
    print(f"{'Filename':<35} | {'Food Prob':<10} | {'EfficientNet Prediction':<20} | {'Confidence':<10}")
    print("-" * 83)
    for r in non_food_results:
        if r["binary_decision"] == "FOOD":
            print(f"{r['filename'][:35]:<35} | {r['food_prob']*100:8.2f}%  | {r['eff_class']:<20} | {r['eff_conf']:6.2f}%")

print("\n" + "-" * 70)
print("C. OVERALL COMBINED PIPELINE COMPARISON SUMMARY:")
print("-" * 70)
print(f"{'Metric / Evaluation Scope':<42} | {'EffNet-B0 Alone':<16} | {'Binary V3 Alone':<16} | {'Combined Pipeline':<18}")
print("-" * 98)
print(f"{'Genuine Thali Accuracy (90 images)':<42} | {eff_alone_correct}/90 ({eff_alone_accuracy:.2f}%)  | {thali_accepted}/90 ({thali_acceptance_rate:.2f}%)  | {overall_thali_correct}/90 ({overall_thali_accuracy:.2f}%)")
print(f"{'Non-Food Rejection Rate (37 images)':<42} | N/A (0.00%)      | {non_food_rejected}/37 ({non_food_rejection_rate:.2f}%) | {non_food_rejected}/37 ({non_food_rejection_rate:.2f}%)")
print(f"{'Non-Food False Acceptance Rate (37 images)':<42} | 100.00% (37/37)  | {non_food_accepted}/37 ({non_food_accepted/total_non_food*100:.2f}%)  | {non_food_accepted}/37 ({non_food_accepted/total_non_food*100:.2f}%)")
print("=" * 98)
