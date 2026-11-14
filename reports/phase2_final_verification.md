# Phase 2 Final Technical Verification Audit: EfficientNet-B0 Baseline

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
**Audit Date:** November 13, 2026
**Auditor:** Antigravity AI Research Verification System
**Audit Scope:** Read-only inspection of Phase 2 source code, data pipelines, checkpoint metadata, test evaluation metrics, and training dynamics.
**Final Status:** **PASS**

---

## 1. Files Inspected

The following codebase artifacts and checkpoints were audited directly:

| File Path | Description | Verified Status |
|---|---|---|
| `src/models/efficientnet_baseline.py` | Architecture definition & initialization | Verified (Strictly Scratch) |
| `src/dataset.py` | Manifest loading, augmentation & transforms | Verified (Zero Leakage) |
| `src/train_efficientnet.py` | Training loop, class weighting & early stopping | Verified (No Test Data Used) |
| `src/evaluate_efficientnet.py` | Post-training evaluation on primary test set | Verified (Correct Metrics) |
| `models/efficientnet_b0_scratch_best.pth` | Saved model & training checkpoint | Verified (Epoch 13 Weights) |
| `reports/phase2_training_history.csv` | Epoch-by-epoch training logs | Verified (25 Epochs logged) |
| `reports/phase2_efficientnet_metrics.json` | Final quantitative test results | Verified (Matches Evaluation) |
| `reports/phase2_confusion_matrix.png` | Saved confusion matrix visualization | Verified (Accurate counts) |
| `reports/phase2_learning_curves.png` | Train vs. Val loss & accuracy curves | Verified (Matches CSV logs) |

---

## 2. Random Initialization Audit

### 2.1 Codebase Inspection (`src/models/efficientnet_baseline.py`)

The architecture constructor strictly instantiates torchvision's `efficientnet_b0` with `weights=None`:

```python
# Lines 31-36 of src/models/efficientnet_baseline.py
self.num_classes = num_classes
self.is_pretrained = False  # Explicit verification flag

# STRICT: weights=None explicitly guarantees no pretrained parameters are loaded
self.backbone = efficientnet_b0(weights=None)
```

The classification head is replaced to output 2 logits (binary oral cancer classification):

```python
# Lines 46-61 of src/models/efficientnet_baseline.py
new_classifier = nn.Sequential(
    nn.Dropout(p=dropout, inplace=True),
    nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
)

# Initialize the newly created linear layer
linear_layer = new_classifier[1]
if init_head == "kaiming":
    nn.init.kaiming_normal_(linear_layer.weight, mode="fan_out", nonlinearity="relu")
elif init_head == "xavier":
    nn.init.xavier_uniform_(linear_layer.weight)
else:
    nn.init.normal_(linear_layer.weight, mean=0.0, std=0.01)
nn.init.constant_(linear_layer.bias, 0.0)

self.backbone.classifier = new_classifier
```

### 2.2 Verification of Layer-Wise Parameter Initialization

An inspection of torchvision's internal `EfficientNet.__init__` implementation confirms that when `weights=None`, weights are initialized according to layer type rather than an arbitrary universal Gaussian:

```python
# torchvision/models/efficientnet.py lines 177-187
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init_range = 1.0 / math.sqrt(m.out_features)
        nn.init.uniform_(m.weight, -init_range, init_range)
        nn.init.zeros_(m.bias)
```

**Layer Type Breakdown:**
1. **2D Convolutional Layers (`nn.Conv2d`):** Initialized using **Kaiming Normal (`kaiming_normal_`, mode='fan_out')**. Biases (where present) are set to $0.0$.
2. **Batch Normalization Layers (`nn.BatchNorm2d`):** Scale parameter (`weight`) initialized to **constant $1.0$** (`nn.init.ones_`), shift parameter (`bias`) initialized to **constant $0.0$** (`nn.init.zeros_`).
3. **Replaced Classifier Head (`nn.Linear`):** Weight tensor initialized using **Kaiming Normal (`kaiming_normal_`, mode='fan_out', nonlinearity='relu')**; bias initialized to **constant $0.0$**.

### 2.3 External Weight & Pretrained Checkpoint Loading Check
- **No external pretrained checkpoints:** No call to `load_state_dict()`, `urllib`, `requests`, `torch.hub`, or external URLs exists in `src/models/` or the initialization pipeline.
- **Model attribute:** `model.is_pretrained == False`.
- **Verdict:** Strict from-scratch initialization is verified.

---

## 3. Training and Data Pipeline Audit

### 3.1 Dataset Splits and Class Mapping (`src/train_efficientnet.py` & `src/dataset.py`)

The pipeline exclusively ingests the approved primary duplicate-collapsed manifests:

```python
# Lines 126-127 of src/train_efficientnet.py
train_csv: str = "data/splits_primary/train.csv",
val_csv: str = "data/splits_primary/val.csv"
```

The sample distribution across splits is strictly partitioned:

| Dataset Split | File Manifest Path | Total Samples | Cancer (Class 1) | Non-Cancer (Class 0) | Cancer Ratio |
|---|---|---|---|---|---|
| **Training** | `data/splits_primary/train.csv` | **486** | 322 | 164 | 66.26% |
| **Validation** | `data/splits_primary/val.csv` | **108** | 73 | 35 | 67.59% |
| **Testing** | `data/splits_primary/test.csv` | **109** | 75 | 34 | 68.81% |
| **Total Primary** | — | **703** | **470** | **233** | 66.86% |

- **Class Mapping:** `0: "NON CANCER"`, `1: "CANCER"`.
- **Zero Scene Leakage:** Verified in Phase 1 that every duplicate group exists strictly within a single split partition, and redundant scene copies were completely excluded.

### 3.2 Normalization and Data Augmentation Audit

1. **Normalization Statistics:**
   ```python
   # Lines 37-38 of src/dataset.py
   DATASET_PRIMARY_MEAN = [0.6432, 0.4537, 0.4275]
   DATASET_PRIMARY_STD  = [0.2189, 0.2118, 0.2090]
   ```
   Computed strictly across the 486 training images in `data/splits_primary/train.csv`. No validation or test image pixels were included.

2. **Training Augmentation:**
   ```python
   # Lines 134-141 of src/dataset.py
   transforms.Compose([
       resizer,
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.RandomRotation(degrees=10, interpolation=transforms.InterpolationMode.BILINEAR),
       transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.0, hue=0.0),
       transforms.ToTensor(),
       transforms.Normalize(mean=mean, std=std),
   ])
   ```
   Conservative augmentations preserving clinical oral geometry:
   - Bilateral horizontal flipping preserves oral anatomical symmetry.
   - 10-degree rotation accounts for patient head angle variation.
   - Minor brightness/contrast jitter ($\pm 5\%$) accounts for camera exposure differences without altering lesion hue/saturation.
   - Vertical flipping is prohibited (anatomically unnatural).

3. **Validation and Test Transformations:**
   ```python
   # Lines 144-148 of src/dataset.py
   transforms.Compose([
       resizer,
       transforms.ToTensor(),
       transforms.Normalize(mean=mean, std=std),
   ])
   ```
   Strictly deterministic: bilinear resize to $224 \times 224$, conversion to tensor, and primary z-score normalization. No random transforms exist for validation or testing.

### 3.3 Loss Weighting and Data Leakage Prevention

- **Class Weight Calculation:** Computed dynamically using only the training manifest labels:
  $$\text{weight}_c = \frac{N_{\text{total}}}{2 \times N_c}$$
  $$\text{weight}_{\text{Non-Cancer (0)}} = \frac{486}{2 \times 164} \approx 1.4817$$
  $$\text{weight}_{\text{Cancer (1)}} = \frac{486}{2 \times 322} \approx 0.7547$$
  Passed into `nn.CrossEntropyLoss(weight=class_weights)`.
- **Test Set Isolation:** The test manifest (`data/splits_primary/test.csv`) is never opened, parsed, or evaluated during training or early stopping checks. It was evaluated strictly once via `src/evaluate_efficientnet.py` post-training.

### 3.4 Reproducibility and DataLoader Settings

- Seed fixed at `42` across `random`, `numpy`, `torch`, and CUDA.
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` enforced.
- `num_workers = 0` was configured for all DataLoaders, precluding multi-process generator fork desynchronization.

---

## 4. Checkpoint Audit (`models/efficientnet_b0_scratch_best.pth`)

### 4.1 Structural Inspection

Inspection of the checkpoint dictionary on disk confirmed the following schema:

```python
{
  'epoch': 13,
  'best_val_loss': 0.9228937360975478,
  'best_val_metrics': {
    'val_loss': 0.9228937360975478,
    'val_acc': 0.8055555555555556,
    'val_prec': 0.8513513513513513,
    'val_rec': 0.863013698630137,
    'val_f1': 0.8571428571428571
  },
  'config': {
    'model': 'EfficientNet-B0 (Scratch)',
    'weights': None,
    'is_pretrained': False,
    'seed': 42,
    'batch_size': 16,
    'lr': 0.0003,
    'weight_decay': 0.01,
    'max_epochs': 50,
    'patience': 12,
    'class_weights': [1.4817073345184326, 0.7546584010124207],
    'norm_mean': [0.6432, 0.4537, 0.4275],
    'norm_std': [0.2189, 0.2118, 0.209],
    'train_manifest': 'data/splits_primary/train.csv',
    'val_manifest': 'data/splits_primary/val.csv',
    'class_mapping': {0: 'NON CANCER', 1: 'CANCER'}
  },
  'env': {
    'python_version': '3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]',
    'torch_version': '2.6.0+cu124',
    'device': 'cuda',
    'device_name': 'NVIDIA GeForce RTX 4050 Laptop GPU'
  },
  'model_state_dict': 360 parameter tensors,
  'optimizer_state_dict': {'state', 'param_groups'},
  'scheduler_state_dict': {'factor': 0.5, 'patience': 4, 'best': 0.9228937360975478, 'last_epoch': 13, ...}
}
```

### 4.2 PyTorch 2.6 `weights_only=False` Requirement

In PyTorch 2.6, the default behavior of `torch.load` was shifted to `weights_only=True` to mitigate arbitrary code execution from untrusted pickles. Under `weights_only=True`, unpickling non-primitive classes (e.g., custom configuration dicts, optimizer state structures, or TorchVersion objects) triggers an `_pickle.UnpicklingError`.

Because `models/efficientnet_b0_scratch_best.pth` was generated locally inside this repository and contains comprehensive audit metadata (`config`, `env`, `scheduler_state_dict`), setting `weights_only=False` in `src/evaluate_efficientnet.py` is safe, standard, and required for deserializing the metadata alongside the model weights.

---

## 5. Metric Audit

### 5.1 Positive Class Identification
- **Class 0:** `NON CANCER` (Negative Class)
- **Class 1:** `CANCER` (Positive Class, Target Condition)

### 5.2 Mathematical Formulation & Verified Values

Test evaluation was conducted on all 109 samples in `data/splits_primary/test.csv`.

```
Confusion Matrix Layout:
                     Predicted Non-Cancer (0)    Predicted Cancer (1)
Actual Non-Cancer (0):      TN = 28                     FP = 6
Actual Cancer (1):          FN = 15                     TP = 60
```

1. **Accuracy:**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{60 + 28}{109} = \frac{88}{109} \approx \mathbf{80.73\%}$$

2. **Sensitivity (Recall on Cancer):**
   $$\text{Sensitivity} = \frac{TP}{TP + FN} = \frac{60}{60 + 15} = \frac{60}{75} = \mathbf{80.00\%}$$

3. **Specificity (Recall on Non-Cancer):**
   $$\text{Specificity} = \frac{TN}{TN + FP} = \frac{28}{28 + 6} = \frac{28}{34} \approx \mathbf{82.35\%}$$

4. **Precision (Cancer):**
   $$\text{Precision} = \frac{TP}{TP + FP} = \frac{60}{60 + 6} = \frac{60}{66} \approx \mathbf{90.91\%}$$

5. **$F_1$-Score (Cancer):**
   $$F_1 = 2 \times \frac{\text{Precision} \times \text{Sensitivity}}{\text{Precision} + \text{Sensitivity}} = 2 \times \frac{0.90909 \times 0.80000}{0.90909 + 0.80000} \approx \mathbf{0.8511}$$

6. **ROC-AUC & PR-AUC Probability Verification:**
   - Evaluated via `sklearn.metrics.roc_auc_score(all_targets, all_probs)` and `sklearn.metrics.average_precision_score(all_targets, all_probs)`.
   - In `src/evaluate_efficientnet.py`, `all_probs` contains continuous Softmax confidence scores for Class 1 (`probs[:, 1]`), ranging continuously between $0.0$ and $1.0$.
   - **ROC-AUC:** $\mathbf{0.8912}$
   - **PR-AUC (Average Precision):** $\mathbf{0.9176}$
   - Neither ROC-AUC nor PR-AUC was computed using binarized discrete predictions, confirming full scientific compliance.

---

## 6. Training Curve & Overfitting Analysis

Analysis of `reports/phase2_training_history.csv` across 25 epochs:

| Epoch Range | Train Loss | Train Acc | Val Loss | Val Acc | Validation Dynamic |
|---|---|---|---|---|---|
| **Epoch 1** | 6.8469 | 56.17% | 0.9857 | 67.59% | Cold start from random initialization |
| **Epochs 2–6** | 3.84 $\rightarrow$ 2.40 | 56.6% $\rightarrow$ 62.3% | 1.43 $\rightarrow$ 69.81 $\rightarrow$ 4.58 | 38.0% $\rightarrow$ 73.1% | High gradient volatility (scratch training on small $N$) |
| **Epochs 7–12** | 1.64 $\rightarrow$ 1.09 | 62.3% $\rightarrow$ 70.8% | 3.39 $\rightarrow$ 8.54 | 66.7% $\rightarrow$ 75.0% | LR dropped to $1.5 \times 10^{-4}$ & $7.5 \times 10^{-5}$; stabilization |
| **Epoch 13 (Best)** | **1.5404** | **74.49%** | **0.9229** | **80.56%** | **Global minimum validation loss; checkpoint saved** |
| **Epochs 14–24** | 1.62 $\rightarrow$ 0.68 | 77.2% $\rightarrow$ 83.5% | 1.89 $\rightarrow$ 5.99 $\rightarrow$ 4.05 | 77.8% $\rightarrow$ 83.3% | Onset of overfitting; val loss diverges above 2.0 |
| **Epoch 25** | 0.6400 | 81.48% | 2.2374 | 81.48% | 12 epochs without improvement; early stopping halts |

### Key Observations:
1. **Validation Loss Minimum at Epoch 13:** The checkpoint `models/efficientnet_b0_scratch_best.pth` represents the exact minimum validation loss ($0.9229$).
2. **Overfitting Emergence:** Beyond Epoch 13, training loss continued to decrease toward $0.6399$ while validation loss remained elevated between $1.89$ and $5.99$. This is the classic signature of over-parameterization when training a 4.0-million parameter convolutional network from scratch on 486 samples.
3. **Early Stopping Integrity:** With patience set to 12, the training loop waited from Epoch 14 through Epoch 25 ($25 - 13 = 12$ epochs) without recording a lower validation loss, terminating cleanly at Epoch 25.

---

## 7. Issues Found & Recommended Improvements

| Item | Finding | Impact | Status / Action |
|---|---|---|---|
| **RNG State in Checkpoint** | Random generator states (`torch.get_rng_state()`) are not saved in the checkpoint dictionary. | Minor. The seed (`42`) is preserved in `config['seed']`, and the run finished without needing mid-run resume. | For future phases, optionally include `rng_state` if mid-training pause/resume is required. |
| **`weights_only` Default** | PyTorch 2.6 raises an unpickling error if `weights_only=True` is used on checkpoints containing rich metadata. | Handled. `weights_only=False` is explicitly used in `src/evaluate_efficientnet.py`. | Standard behavior for internal trusted model checkpoints. |

---

## 8. Final Assessment

### Verdict: **PASS**

**Summary Justification:**
1. **Strictly from-scratch:** Fully verified `weights=None` initialization across all convolutional, batch norm, and classifier layers. Zero pretrained weights were introduced.
2. **Zero data leakage:** Primary duplicate-collapsed manifests (`data/splits_primary/`) strictly maintained scene isolation. Normalization and class weights derived exclusively from the 486 training images.
3. **Valid test evaluation:** The primary test set was evaluated once post-training using continuous Softmax probabilities for AUC computations, achieving **80.73% Accuracy**, **80.00% Sensitivity**, and **82.35% Specificity**.
4. **Reliable checkpointing:** Checkpoint corresponds strictly to the global minimum validation loss at Epoch 13 prior to over-parameterization divergence.
