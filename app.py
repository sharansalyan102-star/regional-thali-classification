import os
import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify

from data.thali_info import THALI_INFO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload limit

# ============================================================
# SETTINGS & PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EFF_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "efficientnet_best.pth")
BINARY_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "binary_food_classifier_v3.pth")

THALI_CLASSES = ["Bengali thali", "Kerala Sadhya", "Rajasthani"]
BINARY_CLASSES = ["food", "non_food"]

FOOD_THRESHOLD = 50.0        # Stage 1 Binary V3 gate threshold (50%)
CONFIDENCE_THRESHOLD = 80.0  # Stage 2 EfficientNet-B0 confidence threshold (80%)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Flask App starting... Using device: {DEVICE}")

# ============================================================
# TRANSFORMS
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
# MODEL LOADING
# ============================================================
binary_model = None
eff_model = None

try:
    # 1. Binary V3 Classifier (Stage 1 Gate)
    if os.path.exists(BINARY_MODEL_PATH):
        binary_model = models.efficientnet_b0(weights=None)
        binary_model.classifier[1] = nn.Linear(binary_model.classifier[1].in_features, len(BINARY_CLASSES))
        binary_ckpt = torch.load(BINARY_MODEL_PATH, map_location=DEVICE)
        binary_state_dict = binary_ckpt.get("model_state_dict", binary_ckpt)
        binary_model.load_state_dict(binary_state_dict)
        binary_model = binary_model.to(DEVICE)
        binary_model.eval()
        print("Successfully loaded Stage-1 Binary V3 Classifier model.")
    else:
        print(f"Error: Binary model not found at {BINARY_MODEL_PATH}")

    # 2. EfficientNet-B0 Classifier (Stage 2 3-Class Model)
    if os.path.exists(EFF_MODEL_PATH):
        eff_model = models.efficientnet_b0(weights=None)
        eff_model.classifier[1] = nn.Linear(eff_model.classifier[1].in_features, len(THALI_CLASSES))
        eff_ckpt = torch.load(EFF_MODEL_PATH, map_location=DEVICE)
        eff_state_dict = eff_ckpt.get("model_state_dict", eff_ckpt)
        eff_model.load_state_dict(eff_state_dict)
        eff_model = eff_model.to(DEVICE)
        eff_model.eval()
        print("Successfully loaded Stage-2 EfficientNet-B0 Thali Classifier model.")
    else:
        print(f"Error: EfficientNet checkpoint not found at {EFF_MODEL_PATH}")
except Exception as e:
    print(f"Error loading models: {e}")


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if binary_model is None or eff_model is None:
        return jsonify({"success": False, "error": "Classification models are not properly loaded."}), 500

    if 'file' not in request.files and 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file provided."}), 400

    file = request.files.get('file') or request.files.get('image')

    if file.filename == '':
        return jsonify({"success": False, "error": "Empty filename."}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = test_transform(image).unsqueeze(0).to(DEVICE)

        # ------------------------------------------------------------
        # STAGE 1: BINARY V3 FOOD / NON-FOOD GATE
        # ------------------------------------------------------------
        with torch.no_grad():
            bin_output = binary_model(img_tensor)
            bin_probs = torch.softmax(bin_output, dim=1)
            food_idx = BINARY_CLASSES.index("food")
            food_prob = bin_probs[0, food_idx].item() * 100.0

        # If food probability < 50%, STOP immediately! Do NOT run EfficientNet-B0!
        if food_prob < FOOD_THRESHOLD:
            print(f"[STAGE-1 REJECTION] Food probability: {food_prob:.2f}% < {FOOD_THRESHOLD}%. Stopping pipeline.")
            return jsonify({
                "success": True,
                "is_food": False,
                "is_low_confidence": False,
                "food_probability": round(food_prob, 2),
                "prediction": "Not a Recognized Thali",
                "message": "Please upload an image of a Bengali Thali, Kerala Sadhya, or Rajasthani Thali."
            })

        # ------------------------------------------------------------
        # STAGE 2: ORIGINAL EFFICIENTNET-B0 3-CLASS CLASSIFIER
        # ------------------------------------------------------------
        print(f"[STAGE-1 PASSED] Food probability: {food_prob:.2f}% >= {FOOD_THRESHOLD}%. Running EfficientNet-B0.")
        with torch.no_grad():
            outputs = eff_model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probabilities, dim=0)

        predicted_class_name = THALI_CLASSES[pred_idx.item()]
        confidence_val = conf.item() * 100.0

        # Construct probability dictionary for all 3 classes
        prob_dict = {
            "Bengali Thali": round(probabilities[0].item() * 100.0, 2),
            "Kerala Sadhya": round(probabilities[1].item() * 100.0, 2),
            "Rajasthani Thali": round(probabilities[2].item() * 100.0, 2)
        }

        # Check if top predicted class confidence is < 80%
        if confidence_val < CONFIDENCE_THRESHOLD:
            print(f"[STAGE-2 LOW CONFIDENCE] Top prediction: '{predicted_class_name}' with {confidence_val:.2f}% < {CONFIDENCE_THRESHOLD}%. Marking as low confidence.")
            return jsonify({
                "success": True,
                "is_food": True,
                "is_low_confidence": True,
                "food_probability": round(food_prob, 2),
                "confidence": round(confidence_val, 2),
                "top_candidate": predicted_class_name,
                "prediction": "Low Classification Confidence",
                "message": "The image may contain food, but the model is not confident enough (confidence < 80%) to identify the regional thali. Please upload a clearer image showing the complete thali.",
                "probabilities": prob_dict
            })

        # High confidence (>= 80%) prediction
        info_data = THALI_INFO.get(predicted_class_name, {})
        return jsonify({
            "success": True,
            "is_food": True,
            "is_low_confidence": False,
            "food_probability": round(food_prob, 2),
            "prediction": info_data.get("title", predicted_class_name),
            "confidence": round(confidence_val, 2),
            "probabilities": prob_dict,
            "thali_info": info_data
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"success": False, "error": f"Failed to process image: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
