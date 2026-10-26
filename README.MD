# 🍛 Regional Thali Classification

A deep learning-based image classification system for identifying **three regional Indian thali categories** from food images:

- 🇮🇳 Bengali Thali
- 🥘 Kerala Sadhya
- 🍱 Rajasthani Thali

The project uses **EfficientNet-B0 with transfer learning** and a **two-stage classification pipeline** to distinguish food images from non-food inputs before performing regional thali classification. The trained models are integrated into a **Flask web application** for real-time image prediction.

---

## 📌 Project Overview

Indian thalis vary significantly across regions in terms of ingredients, presentation, preparation style, and arrangement. Identifying the regional cuisine from an image can be challenging because different thalis may contain visually similar dishes.

This project develops a computer vision system capable of recognizing three selected regional Indian thalis from images.

The system consists of two stages:

```text
                    Input Image
                         │
                         ▼
             ┌──────────────────────┐
             │ Stage 1              │
             │ Food / Non-Food      │
             │ Binary Classifier    │
             └──────────┬───────────┘
                        │
                 Food Probability
                     ≥ 50%?
                    /       \
                  No         Yes
                  │           │
                  ▼           ▼
              Reject     Stage 2
                          EfficientNet-B0
                               │
                               ▼
                  ┌───────────────────────┐
                  │ Regional Classification│
                  ├───────────────────────┤
                  │ Bengali Thali         │
                  │ Kerala Sadhya         │
                  │ Rajasthani Thali      │
                  └───────────┬───────────┘
                              │
                              ▼
                       Confidence Check
                           ≥ 70%
                              │
                              ▼
                       Final Prediction
```

---

## 🎯 Objectives

- Classify images into three regional Indian thali categories.
- Use transfer learning to achieve effective image classification with a relatively limited dataset.
- Prevent obvious non-food images from directly reaching the regional classifier.
- Provide confidence-based handling for uncertain predictions.
- Deploy the trained model through a Flask web interface.

---

## 🧠 Models Used

### Stage 1 — Food / Non-Food Classifier

**Model:** EfficientNet-B0

The first stage determines whether an uploaded image is likely to be food.

The selected Binary V3 model uses a **50% food-probability threshold**.

```text
Food probability < 50%
        ↓
Reject image
```

```text
Food probability ≥ 50%
        ↓
Send image to Stage 2
```

### Stage 2 — Regional Thali Classifier

**Model:** EfficientNet-B0

The second stage classifies the image into:

1. Bengali Thali
2. Kerala Sadhya
3. Rajasthani Thali

A **70% classification confidence threshold** is used for the final prediction.

---

## 📊 Model Performance

### Regional Thali Classifier

The original EfficientNet-B0 model achieved:

**Test Accuracy: 96.67%**

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Bengali Thali | 93.55% | 96.67% | 95.08% |
| Kerala Sadhya | 100.00% | 100.00% | 100.00% |
| Rajasthani Thali | 96.55% | 93.33% | 94.92% |
| **Overall Accuracy** | | | **96.67%** |

### Binary Food Classifier — V3

Evaluation on the project test sets:

- Genuine thali images: **86/90 accepted**
- Genuine food acceptance: **95.56%**
- Non-food images: **32/37 rejected**
- Non-food rejection: **86.49%**

The original V3 model was selected for final integration after comparison with an expanded experiment.

---

## 🔬 Experiments

Several experiments were conducted during development.

### ResNet50

The initial regional classifier achieved:

**94.44% test accuracy**

### EfficientNet-B0

The EfficientNet-B0 model improved the result to:

**96.67% test accuracy**

Therefore, EfficientNet-B0 was selected as the final regional classifier.

### Data Augmentation Experiment

An augmented EfficientNet-B0 experiment was also performed.

The augmented model achieved:

**92.22% test accuracy**

Since this was lower than the original EfficientNet-B0 result, the original model was retained.

### Binary Classifier V3

The Binary V3 classifier was introduced to handle non-food images before regional classification.

The selected model achieved:

- **95.56% genuine-food acceptance**
- **86.49% non-food rejection**

### Expanded Binary Experiment

An additional experiment was performed by adding 70 new non-food images.

The expanded model did not improve performance:

- Non-food rejection: **56.76%**
- Genuine food acceptance: **50.00%**

Therefore, the expanded model was **not integrated into the final application**.

---

## 🗂️ Dataset

The regional classification dataset contains three target classes:

```text
dataset/
├── Bengali thali/
├── Kerala Sadhya/
└── Rajasthani/
```

The dataset was divided into training, validation, and testing subsets.

The final regional classifier was evaluated on a test set containing **90 genuine thali images**.

A separate set of **37 non-food images** was used as a stress-test dataset for evaluating the binary food/non-food classifier.

> The non-food set is treated as a practical stress test rather than a completely independent scientific benchmark because of overlap with data used during binary-classifier development.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Torchvision | Computer vision utilities |
| timm | Pretrained EfficientNet-B0 model |
| PIL | Image processing |
| Flask | Web application |
| NumPy | Numerical processing |
| Matplotlib | Evaluation and visualization |
| Git & GitHub | Version control |

---

## 🔄 Transfer Learning

The project uses a pretrained **EfficientNet-B0** architecture.

Instead of training the complete neural network from scratch, pretrained weights are used and the classification layer is adapted for the project's target classes.

This approach helps when the available dataset is relatively limited.

---

## 🧪 Training Configuration

### Regional Classifier

- Architecture: EfficientNet-B0
- Transfer learning: Yes
- Input size: `224 × 224`
- Optimizer: Adam
- Loss function: Cross Entropy Loss
- Training epochs: 15
- Device: CPU

### Image Preprocessing

Images are resized and normalized using ImageNet normalization.

Training experiments also included:

- RandomResizedCrop
- RandomHorizontalFlip
- RandomRotation
- ColorJitter

---

## 🌐 Flask Web Application

The trained models are integrated into a Flask application.

The application allows a user to:

1. Upload an image.
2. Perform food/non-food detection.
3. Reject images below the food threshold.
4. Classify valid food images into the three regional categories.
5. Apply a confidence threshold to the final prediction.
6. Display the predicted thali category.

---

## 📁 Project Structure

```text
Regional-Thali-Classification/
│
├── checkpoints/
│   ├── efficientnet_best.pth
│   └── binary_food_classifier_v3.pth
│
├── dataset/
│   ├── Bengali thali/
│   ├── Kerala Sadhya/
│   └── Rajasthani/
│
├── outputs/
│   └── confusion_matrix.png
│
├── unknown_test/
│   └── non_food/
│
├── app.py
├── train.py
├── test.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Regional-Thali-Classification.git
cd Regional-Thali-Classification
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Flask application:

```bash
python app.py
```

The application will run locally at:

```text
http://127.0.0.1:5000
```

Open the address in a web browser and upload a thali image.

---

## 📈 Evaluation

The project uses several evaluation measures:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Food acceptance rate
- Non-food rejection rate

The confusion matrix helps identify which regional thali classes are most frequently confused.

---

## ⚠️ Limitations

- The regional classifier recognizes only three trained categories.
- An unseen regional thali may be assigned to one of the known classes if it passes the food gate.
- Performance depends on image quality, lighting, presentation, and similarity to the training data.
- The binary classifier is designed primarily for food/non-food filtering and does not independently identify whether a food image belongs to one of the three target regional cuisines.
- More diverse real-world images can improve robustness.

---

## 🔮 Future Enhancements

- Add more Indian regional thali categories such as Maharashtrian, Karnataka, Andhra, Gujarati, and Punjabi thalis.
- Increase dataset size and diversity.
- Collect more real-world images under different lighting and presentation conditions.
- Improve out-of-distribution detection.
- Explore advanced confidence calibration techniques.
- Deploy the application on a cloud platform.
- Develop a mobile application for real-time thali recognition.
- Add nutritional and ingredient information for each identified thali.

---

## 👨‍💻 Project Highlights

- **96.67% test accuracy** using EfficientNet-B0
- **Two-stage image classification architecture**
- **Food/non-food filtering**
- **Confidence-based prediction**
- **Transfer learning**
- **Flask deployment**
- **Real-world stress testing**
- **Multiple model experiments and comparative evaluation**

---

## 📜 License

This project is developed for academic and educational purposes.

---

## ⭐ Acknowledgements

This project was developed as part of an academic project focused on applying **deep learning and computer vision to Indian regional cuisine recognition**.