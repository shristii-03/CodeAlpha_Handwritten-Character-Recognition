# ✍️ Handwritten Character Recognition

> Identify handwritten **digits (0–9)** using **image processing** and a **Convolutional Neural Network (CNN)** — achieving **99.67% test accuracy** on the MNIST benchmark.

---

## 📌 Project Overview

This project builds a deep learning pipeline to classify handwritten digit images using a custom CNN trained on the MNIST dataset. It covers the complete workflow — from data loading and augmentation to model training, evaluation, and a full visualisation dashboard.

---

## 🗂️ Project Structure

```
CodeAlpha_Handwritten Character Recognition_T2/
│
├── handwritten_recognition.py           # Main script (full pipeline)
├── handwriting_recognition_results.png  # Output dashboard (7-panel visualisation)
├── best_cnn.pth                         # Saved best model weights
├── mnist_data/                          # Auto-downloaded MNIST dataset
└── README.md                            # Project documentation
```

---

## ⚙️ Requirements

- Python 3.8 or higher

Install all dependencies with:

```bash
pip install torch torchvision matplotlib seaborn scikit-learn numpy
```

---

## 🚀 How to Run

```bash
python handwritten_recognition.py
```

**What happens automatically:**
1. MNIST dataset is downloaded into `mnist_data/` (~11 MB, first run only)
2. Data augmentation is applied during training
3. CNN trains for 10 epochs, saving the best weights to `best_cnn.pth`
4. Evaluation metrics are printed in the terminal
5. Dashboard saved as `handwriting_recognition_results.png`

> No manual dataset setup needed — everything downloads automatically!

---

## 📊 Dataset — MNIST

| Property | Details |
|---|---|
| Source | `torchvision.datasets.MNIST` (auto-download) |
| Train samples | 60,000 images |
| Test samples | 10,000 images |
| Image size | 28 × 28 pixels, grayscale |
| Classes | 10 (digits 0–9) |

### 🔧 Preprocessing & Augmentation

| Step | Details |
|---|---|
| Normalisation | Mean = 0.1307, Std = 0.3081 |
| Random Rotation | ±10 degrees (train only) |
| Random Affine | Translate up to 8% (train only) |
| Tensor conversion | `transforms.ToTensor()` |

---

## 🧠 CNN Architecture

```
Input: (Batch, 1, 28, 28)
│
├── Block 1: Conv2d(1→32)  → BatchNorm → ReLU
│            Conv2d(32→32) → BatchNorm → ReLU
│            MaxPool2d(2×2) → Dropout2d(0.25)
│
├── Block 2: Conv2d(32→64) → BatchNorm → ReLU
│            Conv2d(64→64) → BatchNorm → ReLU
│            MaxPool2d(2×2) → Dropout2d(0.25)
│
├── Block 3: Conv2d(64→128) → BatchNorm → ReLU → Dropout2d(0.25)
│
└── Head:    Flatten
             Linear(128×7×7 → 512) → BatchNorm → ReLU → Dropout(0.5)
             Linear(512 → 10)
│
Output: (Batch, 10) class logits
```

| Property | Value |
|---|---|
| Total Parameters | 1,748,458 |
| Activation | ReLU |
| Regularisation | BatchNorm + Dropout |
| Loss Function | CrossEntropyLoss (label smoothing=0.05) |
| Optimiser | Adam (lr=1e-3, weight_decay=1e-4) |
| LR Scheduler | StepLR (step=3, gamma=0.5) |

---

## 📈 Results

| Metric | Score |
|---|---|
| **Test Accuracy** | **99.67%** 🏆 |
| **Macro F1-Score** | **0.9967** |
| **Precision (avg)** | **1.00** |
| **Recall (avg)** | **1.00** |
| Best Val Accuracy | 99.67% |
| Epochs Trained | 10 |

### Per-Class Performance

| Digit | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 | 1.00 | 1.00 | 1.00 |
| 1 | 1.00 | 1.00 | 1.00 |
| 2 | 1.00 | 1.00 | 1.00 |
| 3 | 1.00 | 1.00 | 1.00 |
| 4 | 1.00 | 1.00 | 1.00 |
| 5 | 1.00 | 1.00 | 1.00 |
| 6 | 1.00 | 1.00 | 1.00 |
| 7 | 1.00 | 1.00 | 1.00 |
| 8 | 1.00 | 1.00 | 1.00 |
| 9 | 1.00 | 1.00 | 1.00 |

---

## 🖼️ Visualisation Dashboard

Running the script generates a **7-panel figure** (`handwriting_recognition_results.png`):

| Panel | Description |
|---|---|
| 📉 Loss Curve | Train vs Validation loss per epoch |
| 📈 Accuracy Curve | Train vs Validation accuracy per epoch |
| 📋 Model Summary Card | Key metrics and model stats at a glance |
| 🖼️ Sample Predictions | 36 test images — green = correct, red = wrong |
| 🔲 Confusion Matrix | 10×10 matrix of True vs Predicted labels |
| 📊 Per-Class Accuracy | Horizontal bar chart for each digit class |
| 🟢 Confidence Distribution | Softmax score distribution for correct vs incorrect predictions |

---

## 🔁 Extending to CRNN (Word/Sentence Recognition)

This CNN is the backbone for a **CRNN (Convolutional Recurrent Neural Network)** — used for full word or sentence recognition:

```
CNN (feature extractor)
    ↓
Sequence of feature columns
    ↓
Bidirectional LSTM (sequence modelling)
    ↓
CTC Loss (Connectionist Temporal Classification)
    ↓
Decoded text output
```

To extend, replace the classification head with a BiLSTM layer and use CTC decoding on datasets like **IAM Handwriting** or **EMNIST Letters**.

---

## 📚 Concepts Covered

- Convolutional Neural Networks (CNN)
- Batch Normalisation & Dropout (regularisation)
- Data Augmentation (rotation, affine transform)
- Learning Rate Scheduling (StepLR)
- CrossEntropyLoss with Label Smoothing
- Model checkpointing (`torch.save` / `torch.load`)
- Evaluation: Accuracy, Precision, Recall, F1-Score
- Confusion Matrix & Confidence Score Analysis
- Extendable to CRNN for sequence recognition

---

## 👤 Author
Task 3  — Handwritten Character Recognition** .
