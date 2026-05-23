"""
=============================================================
  TASK 3: Handwritten Character Recognition
  Objective : Identify handwritten characters / alphabets
  Approach  : Image processing + Deep Learning (CNN)
  Dataset   : MNIST digits (0–9) via torchvision auto-download
  Model     : Custom CNN  |  Extendable to CRNN
=============================================================
"""

import os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 62)
print("  TASK 3 — Handwritten Character Recognition (CNN)")
print("=" * 62)
print(f"\n  Device : {DEVICE}")

# ─────────────────────────────────────────────────────────────
# 1.  DATA LOADING  ─  auto-download MNIST
# ─────────────────────────────────────────────────────────────
DATA_DIR = "./mnist_data"          # saved in the same folder as this script
os.makedirs(DATA_DIR, exist_ok=True)

print("\n📥 Downloading / loading MNIST dataset …")

transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform_train)
test_dataset  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform_test)

BATCH        = 256
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH, shuffle=False, num_workers=0)

NUM_CLASSES  = 10
CLASS_NAMES  = [str(i) for i in range(10)]

print(f"  Train samples : {len(train_dataset):,}")
print(f"  Test  samples : {len(test_dataset):,}")
print(f"  Classes       : {NUM_CLASSES}  (digits 0–9)")

# ─────────────────────────────────────────────────────────────
# 2.  CNN ARCHITECTURE
# ─────────────────────────────────────────────────────────────
class HandwritingCNN(nn.Module):
    """
    4-block CNN for handwritten digit recognition.

    Input  : (B, 1, 28, 28)
    Output : (B, 10)

    Architecture
    ────────────
    Block 1 : Conv(32) BN ReLU → Conv(32) BN ReLU → MaxPool → Dropout2d(0.25)
    Block 2 : Conv(64) BN ReLU → Conv(64) BN ReLU → MaxPool → Dropout2d(0.25)
    Block 3 : Conv(128) BN ReLU → Dropout2d(0.25)
    Head    : Flatten → Linear(512) BN ReLU Dropout(0.5) → Linear(10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout2d(0.25)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


model     = HandwritingCNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n🧠 CNN Architecture — {params:,} trainable parameters")
print(model)

# ─────────────────────────────────────────────────────────────
# 3.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────
EPOCHS  = 10
history = {"train_loss": [], "train_acc": [],
           "val_loss":   [], "val_acc":   []}

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if train:
                optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total

print(f"\n🏋️  Training CNN for {EPOCHS} epochs …\n")
best_val  = 0.0
save_path = "best_cnn.pth"     # saved next to the script

for ep in range(1, EPOCHS + 1):
    t0 = time.time()
    tl, ta = run_epoch(train_loader, train=True)
    vl, va = run_epoch(test_loader,  train=False)
    scheduler.step()

    history["train_loss"].append(tl);  history["train_acc"].append(ta)
    history["val_loss"].append(vl);    history["val_acc"].append(va)

    flag = "  ◀ best" if va > best_val else ""
    if va > best_val:
        best_val = va
        torch.save(model.state_dict(), save_path)

    print(f"  Epoch {ep:02d}/{EPOCHS}  "
          f"Train Loss={tl:.4f} Acc={ta*100:.2f}%  |  "
          f"Val Loss={vl:.4f} Acc={va*100:.2f}%  "
          f"[{time.time()-t0:.1f}s]{flag}")

# ─────────────────────────────────────────────────────────────
# 4.  EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n📋 Evaluating best model …")
model.load_state_dict(torch.load(save_path, map_location=DEVICE))
model.eval()

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        out   = model(imgs.to(DEVICE))
        probs = F.softmax(out, dim=1)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

acc = accuracy_score(all_labels, all_preds)
f1  = f1_score(all_labels, all_preds, average='macro')

print(f"\n  ✅ Final Test Accuracy : {acc*100:.2f}%")
print(f"  ✅ Macro F1-Score      : {f1:.4f}")
print(f"  ✅ Best Val Accuracy   : {best_val*100:.2f}%")

print("\n" + "=" * 62)
print("  CLASSIFICATION REPORT — MNIST Digits (0–9)")
print("=" * 62)
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────────────────────
# 5.  VISUALISATION DASHBOARD
# ─────────────────────────────────────────────────────────────
print("\n📈 Generating evaluation dashboard …")

# Collect raw (un-normalised) test images for display
raw_imgs, raw_lbls = [], []
plain_loader = DataLoader(
    datasets.MNIST(DATA_DIR, train=False, download=False,
                   transform=transforms.ToTensor()),
    batch_size=256, shuffle=False, num_workers=0
)
for imgs, lbls in plain_loader:
    raw_imgs.extend(imgs.squeeze().numpy())
    raw_lbls.extend(lbls.numpy())
raw_imgs = np.array(raw_imgs)

plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(22, 24))
fig.patch.set_facecolor('#F0F3F8')
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.38)

# ── A) Loss curve ────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(history["train_loss"], 'o-', color='#2980B9', label='Train', lw=2)
ax0.plot(history["val_loss"],   's-', color='#E74C3C', label='Val',   lw=2)
ax0.set_title('Loss Curve', fontweight='bold')
ax0.set_xlabel('Epoch'); ax0.set_ylabel('Loss')
ax0.legend(); ax0.grid(alpha=0.3)

# ── B) Accuracy curve ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot([a*100 for a in history["train_acc"]], 'o-', color='#27AE60', label='Train', lw=2)
ax1.plot([a*100 for a in history["val_acc"]],   's-', color='#8E44AD', label='Val',   lw=2)
ax1.set_title('Accuracy Curve', fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy (%)')
ax1.legend(); ax1.grid(alpha=0.3)

# ── C) Summary card ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
cards = [
    ("Test Accuracy",  f"{acc*100:.2f}%",      '#27AE60'),
    ("Macro F1-Score", f"{f1:.4f}",             '#2980B9'),
    ("Best Val Acc",   f"{best_val*100:.2f}%",  '#8E44AD'),
    ("Epochs Trained", str(EPOCHS),             '#E67E22'),
    ("Classes",        "10  (digits 0–9)",      '#E74C3C'),
    ("Parameters",     f"{params:,}",           '#16A085'),
]
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.set_title('Model Summary', fontweight='bold', fontsize=12)
for i, (lbl, val, col) in enumerate(cards):
    y = 0.88 - i * 0.15
    ax2.text(0.05, y, f"{lbl}:", fontsize=10, color='#555', va='center')
    ax2.text(0.95, y, val, fontsize=11, color=col, va='center',
             ha='right', fontweight='bold')

# ── D) 36 sample predictions ─────────────────────────────────
inner = gridspec.GridSpecFromSubplotSpec(3, 12, subplot_spec=gs[1, :],
                                          hspace=0.15, wspace=0.1)
np.random.seed(7)
sidx = np.random.choice(len(raw_imgs), 36, replace=False)
for k, idx in enumerate(sidx):
    ax = fig.add_subplot(inner[k // 12, k % 12])
    ax.imshow(raw_imgs[idx], cmap='gray')
    pred = all_preds[idx]
    true = all_labels[idx]
    ax.set_title(f"P:{pred}", fontsize=8,
                 color='green' if pred == true else 'red',
                 fontweight='bold', pad=1)
    ax.axis('off')
fig.text(0.5, 0.595,
         'Sample Predictions  (Green = Correct, Red = Wrong)',
         ha='center', fontsize=12, fontweight='bold')

# ── E) Confusion matrix ──────────────────────────────────────
ax_cm = fig.add_subplot(gs[2, :2])
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=.4, cbar=False)
ax_cm.set_title('Confusion Matrix — Digits 0–9', fontweight='bold')
ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')

# ── F) Per-class accuracy ────────────────────────────────────
ax_pc = fig.add_subplot(gs[2, 2])
per_acc = [(all_preds[all_labels == c] == c).mean() for c in range(NUM_CLASSES)]
bar_colors = ['#2ECC71' if a >= 0.97 else
              '#E67E22' if a >= 0.93 else '#E74C3C' for a in per_acc]
ax_pc.barh(CLASS_NAMES, per_acc, color=bar_colors, edgecolor='white', height=0.65)
ax_pc.set_title('Per-Class Accuracy', fontweight='bold')
ax_pc.set_xlabel('Accuracy'); ax_pc.set_xlim(0, 1.10)
ax_pc.axvline(0.97, color='black', lw=1, ls='--', alpha=0.4)
for i, v in enumerate(per_acc):
    ax_pc.text(v + 0.005, i, f"{v*100:.1f}%", va='center', fontsize=9)

# ── G) Confidence distribution ───────────────────────────────
ax_conf = fig.add_subplot(gs[3, :])
conf_ok  = [all_probs[i, all_labels[i]]
            for i in range(len(all_labels)) if all_preds[i] == all_labels[i]]
conf_bad = [all_probs[i, all_preds[i]]
            for i in range(len(all_labels)) if all_preds[i] != all_labels[i]]
ax_conf.hist(conf_ok,  bins=50, color='#2ECC71', alpha=0.75,
             label=f'Correct ({len(conf_ok):,})',   density=True)
ax_conf.hist(conf_bad, bins=50, color='#E74C3C', alpha=0.75,
             label=f'Incorrect ({len(conf_bad):,})', density=True)
ax_conf.set_title('Prediction Confidence Distribution\n'
                   'Correct vs Incorrect Predictions', fontweight='bold')
ax_conf.set_xlabel('Softmax Confidence Score')
ax_conf.set_ylabel('Density')
ax_conf.legend(fontsize=11)
ax_conf.axvline(0.5, color='black', lw=1.5, ls='--', alpha=0.5)
ax_conf.grid(alpha=0.3)

fig.suptitle('TASK 3 — Handwritten Character Recognition\nCNN Evaluation Dashboard',
             fontsize=17, fontweight='bold', y=1.002)

# Save in the same folder as the script
out_img = 'handwriting_recognition_results.png'
plt.savefig(out_img, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\n   Dashboard saved → {out_img}")
print("\n✅ Task 3 complete!")
print(f"   Test Accuracy : {acc*100:.2f}%  |  Macro F1 : {f1:.4f}")
