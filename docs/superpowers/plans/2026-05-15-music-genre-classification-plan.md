# Music Genre Classification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 16-class music genre classifier using AST (Audio Spectrogram Transformer) fine-tuned from AudioSet, with a ResNet50 baseline for comparison, plus a local mp3 inference pipeline.

**Architecture:** Two models share the same data pipeline (mel spectrogram JPEGs from HF datasets → transforms → DataLoader). AST uses HuggingFace transformers with AudioSet pretrained weights, ResNet50 uses timm with ImageNet weights. Training via a shared Trainer class, evaluation via sklearn metrics + confusion matrices, inference via librosa → mel → model.

**Tech Stack:** PyTorch 2.x, HuggingFace transformers + datasets, timm, librosa, scikit-learn, TensorBoard

---

### Task 1: Project Scaffold

**Files:**
- Create: `D:/music_model_train/.gitignore`
- Create: `D:/music_model_train/requirements.txt`
- Create: directory structure

- [ ] **Step 1: Create .gitignore**

Write `D:/music_model_train/.gitignore`:

```
data/
checkpoints/
logs/
__pycache__/
*.pyc
.DS_Store
*.pt
*.pth
.venv/
venv/
```

- [ ] **Step 2: Create requirements.txt**

Write `D:/music_model_train/requirements.txt`:

```
torch>=2.0.0
torchvision>=0.15.0
datasets>=3.0.0
transformers>=4.40.0
timm>=0.9.0
librosa>=0.10.0
soundfile>=0.12.0
scikit-learn>=1.3.0
tensorboard>=2.14.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.66.0
```

- [ ] **Step 3: Create directory structure**

Run:
```bash
mkdir -p D:/music_model_train/src/data
mkdir -p D:/music_model_train/src/models
mkdir -p D:/music_model_train/src/training
mkdir -p D:/music_model_train/src/evaluation
mkdir -p D:/music_model_train/src/inference
mkdir -p D:/music_model_train/scripts
mkdir -p D:/music_model_train/data
mkdir -p D:/music_model_train/checkpoints
mkdir -p D:/music_model_train/logs
```

- [ ] **Step 4: Create `__init__.py` files**

Run:
```bash
touch D:/music_model_train/src/__init__.py
touch D:/music_model_train/src/data/__init__.py
touch D:/music_model_train/src/models/__init__.py
touch D:/music_model_train/src/training/__init__.py
touch D:/music_model_train/src/evaluation/__init__.py
touch D:/music_model_train/src/inference/__init__.py
```

- [ ] **Step 5: Initialize git and make initial commit**

Run:
```bash
cd D:/music_model_train && git init && git add -A && git commit -m "chore: project scaffold"
```

---

### Task 2: Configuration Module

**Files:**
- Create: `D:/music_model_train/src/training/config.py`

- [ ] **Step 1: Write config.py**

Write `D:/music_model_train/src/training/config.py`:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """AST fine-tuning configuration."""
    model_name: str = "MIT/ast-finetuned-audioset-10-epochs-0.4593"
    num_classes: int = 16
    image_size: int = 224

    batch_size: int = 32
    epochs: int = 30
    lr_backbone: float = 5e-5
    lr_head: float = 5e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    warmup_epochs: int = 5
    early_stop_patience: int = 5

    num_workers: int = 4

    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")

    use_specaugment: bool = True
    time_mask_param: int = 10
    freq_mask_param: int = 2
    use_mixup: bool = True
    mixup_alpha: float = 0.5


@dataclass
class BaselineConfig:
    """ResNet50 baseline configuration."""
    model_name: str = "resnet50"
    num_classes: int = 16
    image_size: int = 224

    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 5

    num_workers: int = 4

    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")

    use_specaugment: bool = True
    time_mask_param: int = 10
    freq_mask_param: int = 2


# Genre label names (Level 3, 16 classes, 0-indexed order)
GENRE_NAMES = [
    "Symphony",
    "Opera",
    "Solo",
    "Chamber",
    "Soul_or_r_and_b",
    "Pop_vocal_ballad",
    "Adult_contemporary",
    "Teen_pop",
    "Contemporary_dance_pop",
    "Dance_pop",
    "Classic_indie_pop",
    "Chamber_cabaret_and_art_pop",
    "Adult_alternative_rock",
    "Uplifting_anthemic_rock",
    "Soft_rock",
    "Acoustic_pop",
]
```

- [ ] **Step 2: Commit**

```bash
cd D:/music_model_train && git add src/training/config.py && git commit -m "feat: add training configuration module"
```

---

### Task 3: Data Module

**Files:**
- Create: `D:/music_model_train/src/data/dataset.py`
- Create: `D:/music_model_train/src/data/transforms.py`

- [ ] **Step 1: Write dataset.py**

Write `D:/music_model_train/src/data/dataset.py`:

```python
from torch.utils.data import DataLoader
from datasets import load_dataset


def get_dataloaders(batch_size: int = 32, num_workers: int = 4):
    """Load CCMUSIC dataset and return train/val/test DataLoaders.

    The dataset provides spectrogram images as PIL Images.
    Training and validation DataLoaders are returned with their
    raw data; transforms are applied in the training loop via
    the transforms module to allow per-batch augmentations (Mixup).
    """
    dataset = load_dataset("ccmusic-database/music_genre")

    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    return train_data, val_data, test_data


def get_label_names(train_data):
    """Return list of 16 genre names from the dataset's ClassLabel feature."""
    return train_data.features["thr_level_label"].names


def collate_fn(batch):
    """Custom collate: extract mel images and thr_level_label from raw HF samples."""
    import torch
    from PIL import Image

    images = []
    labels = []
    for item in batch:
        mel_img = item["mel"]
        if not isinstance(mel_img, Image.Image):
            mel_img = mel_img.convert("RGB")
        images.append(mel_img)
        labels.append(item["thr_level_label"])

    return images, torch.tensor(labels, dtype=torch.long)
```

- [ ] **Step 2: Write transforms.py**

Write `D:/music_model_train/src/data/transforms.py`:

```python
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


def get_train_transforms(image_size: int = 224):
    """Transforms for training: resize, to tensor, normalize to [-1, 1]."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_eval_transforms(image_size: int = 224):
    """Transforms for validation/test: no augmentation."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


class SpecAugment:
    """Frequency and time masking for spectrogram augmentation."""

    def __init__(self, time_mask_param: int = 10, freq_mask_param: int = 2):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, H, W)
        B, C, H, W = x.shape
        # Frequency masking
        for _ in range(self.freq_mask_param):
            f = np.random.randint(0, max(1, H // 10))
            f0 = np.random.randint(0, H - f)
            x[:, :, f0 : f0 + f, :] = 0.0
        # Time masking
        t = int(W * self.time_mask_param / 100)
        t0 = np.random.randint(0, max(1, W - t))
        x[:, :, :, t0 : t0 + t] = 0.0
        return x


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.5):
    """Apply Mixup: blend two samples and their labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    """Compute Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

- [ ] **Step 3: Commit**

```bash
cd D:/music_model_train && git add src/data/ && git commit -m "feat: add data loading and transform modules"
```

---

### Task 4: AST Model

**Files:**
- Create: `D:/music_model_train/src/models/ast_model.py`

- [ ] **Step 1: Write ast_model.py**

Write `D:/music_model_train/src/models/ast_model.py`:

```python
import torch
import torch.nn as nn
from transformers import ASTForAudioClassification


def build_ast_model(model_name: str, num_classes: int, device: torch.device):
    """Load AST model with AudioSet pretrained weights, replace classifier head.

    Args:
        model_name: HuggingFace model ID (e.g. "MIT/ast-finetuned-audioset-10-epochs-0.4593")
        num_classes: Number of target classes (16)
        device: torch device

    Returns:
        model: AST model with replaced classification head
    """
    model = ASTForAudioClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    return model


def get_parameter_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Split parameters into backbone (low LR) and classifier head (high LR) groups."""
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
    ]
```

- [ ] **Step 2: Commit**

```bash
cd D:/music_model_train && git add src/models/ast_model.py && git commit -m "feat: add AST model builder"
```

---

### Task 5: ResNet50 Baseline Model

**Files:**
- Create: `D:/music_model_train/src/models/resnet_baseline.py`

- [ ] **Step 1: Write resnet_baseline.py**

Write `D:/music_model_train/src/models/resnet_baseline.py`:

```python
import torch
import torch.nn as nn
import timm


def build_resnet50(num_classes: int, device: torch.device):
    """Build ResNet50 with ImageNet pretrained weights, 1-channel input, 16-class head.

    Args:
        num_classes: Number of target classes (16)
        device: torch device

    Returns:
        model: ResNet50 model
    """
    model = timm.create_model(
        "resnet50",
        pretrained=True,
        num_classes=num_classes,
        in_chans=1,
    )
    model.to(device)
    return model
```

- [ ] **Step 2: Commit**

```bash
cd D:/music_model_train && git add src/models/resnet_baseline.py && git commit -m "feat: add ResNet50 baseline model"
```

---

### Task 6: Trainer

**Files:**
- Create: `D:/music_model_train/src/training/trainer.py`

- [ ] **Step 1: Write trainer.py**

Write `D:/music_model_train/src/training/trainer.py`:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
import copy

from src.data.transforms import (
    get_train_transforms,
    get_eval_transforms,
    SpecAugment,
    mixup_data,
    mixup_loss,
    collate_fn,
)


def _get_logits(output):
    """Extract logits tensor from model output (HF returns object with .logits, timm returns tensor directly)."""
    return output.logits if hasattr(output, "logits") else output


class Trainer:
    """Generic trainer for music genre classification models."""

    def __init__(
        self,
        model: nn.Module,
        train_data,
        val_data,
        config,
        device: torch.device,
        use_mixup: bool = True,
    ):
        self.model = model
        self.device = device
        self.config = config
        self.use_mixup = use_mixup

        self.train_transform = get_train_transforms(config.image_size)
        self.eval_transform = get_eval_transforms(config.image_size)
        self.specaug = SpecAugment(
            config.time_mask_param, config.freq_mask_param
        ) if getattr(config, 'use_specaugment', False) else None

        self.train_data = train_data
        self.val_data = val_data

        self.writer = SummaryWriter(log_dir=str(config.log_dir))
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_no_improve = 0

    def _transform_batch(self, images, transform):
        """Apply transform to a list of PIL images, return tensor (B, 1, H, W)."""
        batch = torch.stack([transform(img) for img in images])
        return batch

    def _train_epoch(self, optimizer, criterion, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        indices = np.random.permutation(len(self.train_data))
        batch_size = self.config.batch_size

        pbar = tqdm(range(0, len(indices), batch_size), desc=f"Train Epoch {epoch}")
        for start in pbar:
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]
            batch_items = [self.train_data[int(i)] for i in batch_indices]
            images, labels = collate_fn(batch_items)

            x = self._transform_batch(images, self.train_transform)
            labels = labels.to(self.device)
            x = x.to(self.device)

            # SpecAugment
            if self.specaug is not None:
                x = self.specaug(x)

            # Mixup
            if self.use_mixup:
                x, y_a, y_b, lam = mixup_data(x, labels, alpha=self.config.mixup_alpha)
                logits = _get_logits(self.model(x))
                loss = mixup_loss(criterion, logits, y_a, y_b, lam)
                preds = logits.argmax(dim=1)
                total_correct += (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum().item()
            else:
                logits = _get_logits(self.model(x))
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()

            total_samples += len(labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{total_correct / total_samples:.4f}"})

        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def _validate_epoch(self, criterion, epoch):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        batch_size = self.config.batch_size
        pbar = tqdm(range(0, len(self.val_data), batch_size), desc=f"Val Epoch {epoch}")
        for start in pbar:
            end = min(start + batch_size, len(self.val_data))
            batch_items = [self.val_data[int(i)] for i in range(start, end)]
            images, labels = collate_fn(batch_items)

            x = self._transform_batch(images, self.eval_transform)
            x = x.to(self.device)
            labels = labels.to(self.device)

            logits = _get_logits(self.model(x))
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * len(labels)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

        val_loss = total_loss / total_samples
        val_acc = total_correct / total_samples
        return val_loss, val_acc

    def train(self, optimizer, criterion, scheduler=None):
        patience = self.config.early_stop_patience

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self._train_epoch(optimizer, criterion, epoch)
            val_loss, val_acc = self._validate_epoch(criterion, epoch)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Acc/train", train_acc, epoch)
            self.writer.add_scalar("Acc/val", val_acc, epoch)

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
                torch.save(
                    self.best_model_state,
                    self.config.checkpoint_dir / "best_model.pt",
                )
                print(f"  -> Best model saved (val_loss={val_loss:.4f})")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.writer.close()
        # Restore best weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
```

- [ ] **Step 2: Commit**

```bash
cd D:/music_model_train && git add src/training/trainer.py && git commit -m "feat: add training loop with early stopping"
```

---

### Task 7: Evaluation Module

**Files:**
- Create: `D:/music_model_train/src/evaluation/metrics.py`
- Create: `D:/music_model_train/src/evaluation/compare.py`

- [ ] **Step 1: Write metrics.py**

Write `D:/music_model_train/src/evaluation/metrics.py`:

```python
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data.transforms import get_eval_transforms, collate_fn


@torch.no_grad()
def evaluate_model(model, test_data, config, device, label_names=None):
    """Run full evaluation: accuracy, macro/weighted F1, per-class report, confusion matrix.

    Returns:
        dict with keys: accuracy, macro_f1, weighted_f1, per_class (list of dicts),
                        confusion_matrix (np.ndarray), all_preds, all_labels
    """
    model.eval()
    transform = get_eval_transforms(config.image_size)

    all_preds = []
    all_labels = []

    batch_size = config.batch_size
    for start in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        end = min(start + batch_size, len(test_data))
        batch_items = [test_data[int(i)] for i in range(start, end)]
        images, labels = collate_fn(batch_items)

        x = torch.stack([transform(img) for img in images]).to(device)
        output = model(x)
        logits = output.logits if hasattr(output, "logits") else output
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    # Per-class metrics
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        all_labels, all_preds, zero_division=0
    )
    per_class = []
    for i in range(len(precisions)):
        name = label_names[i] if label_names else str(i)
        per_class.append({
            "label": name,
            "precision": round(precisions[i], 4),
            "recall": round(recalls[i], 4),
            "f1": round(f1s[i], 4),
            "support": int(supports[i]),
        })

    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def compute_mid_level_accuracy(all_preds, all_labels, label_names=None):
    """Roll up 16 fine-grained predictions to 9 mid-level classes and compute accuracy.

    Mid-level mapping (index -> mid-level index):
    0-3: Classic (Symphony, Opera, Solo, Chamber) -> mid 0
    4: Soul/R&B -> mid 1
    5-7: Pop (Pop_vocal_ballad, Adult_contemporary, Teen_pop) -> mid 2
    8-9: Dance (Contemporary_dance_pop, Dance_pop) -> mid 3
    10-11: Indie (Classic_indie_pop, Chamber_cabaret_and_art_pop) -> mid 4
    12-15: Rock (Adult_alternative_rock, Uplifting_anthemic_rock, Soft_rock, Acoustic_pop) -> mid 5
    Note: The dataset lists 9 mid-level classes. The above covers 6 families due to the
    fine-grained labels collapsing. Full 9-class rollup needs the sec_level_label mapping.
    For simplicity we use the dataset's own hierarchy.
    """
    # Map each fine-grained label to its parent fine-grained index
    # Uses the dataset's thr_level_label which already encodes hierarchy
    fine_to_mid = {
        0: 0, 1: 1, 2: 2, 3: 3,  # Classic family
        4: 4,                      # Soul/R&B
        5: 5, 6: 5, 7: 5,         # Pop
        8: 6, 9: 6,               # Dance
        10: 7, 11: 7,             # Indie
        12: 8, 13: 8, 14: 8, 15: 8,  # Rock
    }
    mid_preds = np.array([fine_to_mid.get(p, -1) for p in all_preds])
    mid_labels = np.array([fine_to_mid.get(l, -1) for l in all_labels])
    valid = mid_labels >= 0
    return accuracy_score(mid_labels[valid], mid_preds[valid])


def plot_confusion_matrix(cm, label_names, title="Confusion Matrix", save_path=None):
    """Plot and optionally save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.colorbar(im)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")

    plt.close()
    return fig


def print_eval_report(results, label_names):
    """Print formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Accuracy:     {results['accuracy']:.4f}")
    print(f"Macro F1:     {results['macro_f1']:.4f}")
    print(f"Weighted F1:  {results['weighted_f1']:.4f}")
    print(f"\nPer-class metrics:")
    print(f"{'Label':<35} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Support':>7}")
    print("-" * 65)
    for item in results["per_class"]:
        print(f"{item['label']:<35} {item['precision']:>6.4f} {item['recall']:>6.4f} "
              f"{item['f1']:>6.4f} {item['support']:>7}")

    mid_acc = compute_mid_level_accuracy(
        results["all_preds"], results["all_labels"], label_names
    )
    print(f"\nMid-level (9-class rollup) Accuracy: {mid_acc:.4f}")
    print(f"{'='*60}\n")
```

- [ ] **Step 2: Write compare.py**

Write `D:/music_model_train/src/evaluation/compare.py`:

```python
"""AST vs ResNet50 comparison."""
import torch
from src.evaluation.metrics import evaluate_model, plot_confusion_matrix
import matplotlib.pyplot as plt


def compare_models(ast_model, resnet_model, test_data, ast_cfg, baseline_cfg, device, label_names):
    """Evaluate both models side-by-side, print comparison, save confusion matrices."""
    print("\n>>> Evaluating AST model...")
    ast_results = evaluate_model(ast_model, test_data, ast_cfg, device, label_names)

    print("\n>>> Evaluating ResNet50 baseline...")
    resnet_results = evaluate_model(resnet_model, test_data, baseline_cfg, device, label_names)

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'AST':>12} {'ResNet50':>12}")
    print("-" * 48)
    for metric in ["accuracy", "macro_f1", "weighted_f1"]:
        print(f"{metric:<20} {ast_results[metric]:>12.4f} {resnet_results[metric]:>12.4f}")

    mid_ast = __import__("src.evaluation.metrics", fromlist=["compute_mid_level_accuracy"]).compute_mid_level_accuracy(
        ast_results["all_preds"], ast_results["all_labels"], label_names
    )
    mid_resnet = __import__("src.evaluation.metrics", fromlist=["compute_mid_level_accuracy"]).compute_mid_level_accuracy(
        resnet_results["all_preds"], resnet_results["all_labels"], label_names
    )
    print(f"{'Mid-level Acc':<20} {mid_ast:>12.4f} {mid_resnet:>12.4f}")

    # Save confusion matrices
    plot_confusion_matrix(
        ast_results["confusion_matrix"], label_names,
        title="AST Confusion Matrix", save_path="./logs/ast_confusion_matrix.png"
    )
    plot_confusion_matrix(
        resnet_results["confusion_matrix"], label_names,
        title="ResNet50 Confusion Matrix", save_path="./logs/resnet_confusion_matrix.png"
    )

    # Delta bar chart
    per_class_diff = []
    for ast_item, resnet_item in zip(ast_results["per_class"], resnet_results["per_class"]):
        per_class_diff.append({
            "label": ast_item["label"],
            "ast_f1": ast_item["f1"],
            "resnet_f1": resnet_item["f1"],
            "delta": ast_item["f1"] - resnet_item["f1"],
        })

    return {
        "ast": ast_results,
        "resnet": resnet_results,
        "per_class_diff": per_class_diff,
    }
```

- [ ] **Step 3: Commit**

```bash
cd D:/music_model_train && git add src/evaluation/ && git commit -m "feat: add evaluation metrics and model comparison"
```

---

### Task 8: AST Training Script

**Files:**
- Create: `D:/music_model_train/scripts/train_ast.py`

- [ ] **Step 1: Write train_ast.py**

Write `D:/music_model_train/scripts/train_ast.py`:

```python
"""Train AST model for 16-class music genre classification."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.training.config import TrainConfig
from src.data.dataset import get_dataloaders
from src.models.ast_model import build_ast_model, get_parameter_groups
from src.training.trainer import Trainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = TrainConfig()

    # Load data
    print("Loading dataset...")
    train_data, val_data, test_data = get_dataloaders(
        batch_size=config.batch_size, num_workers=config.num_workers
    )

    # Build model
    print("Building AST model...")
    model = build_ast_model(config.model_name, config.num_classes, device)

    # Optimizer with differential learning rates
    param_groups = get_parameter_groups(
        model, config.lr_backbone, config.lr_head, config.weight_decay
    )
    optimizer = AdamW(param_groups)

    # Scheduler: cosine annealing with warmup handled by scheduler internals
    total_steps = (len(train_data) // config.batch_size) * config.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs - config.warmup_epochs)

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    # Train
    trainer = Trainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=config,
        device=device,
        use_mixup=config.use_mixup,
    )

    print("Starting training...")
    trainer.train(optimizer, criterion, scheduler=scheduler)

    print(f"Training complete. Best val_loss: {trainer.best_val_loss:.4f}")
    print(f"Best model saved to {config.checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd D:/music_model_train && git add scripts/train_ast.py && git commit -m "feat: add AST training entry script"
```

---

### Task 9: Baseline Training Script

**Files:**
- Create: `D:/music_model_train/scripts/train_baseline.py`

- [ ] **Step 1: Write train_baseline.py**

Write `D:/music_model_train/scripts/train_baseline.py`:

```python
"""Train ResNet50 baseline for 16-class music genre classification."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.training.config import BaselineConfig
from src.data.dataset import get_dataloaders
from src.models.resnet_baseline import build_resnet50
from src.training.trainer import Trainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = BaselineConfig()

    # Load data
    print("Loading dataset...")
    train_data, val_data, test_data = get_dataloaders(
        batch_size=config.batch_size, num_workers=config.num_workers
    )

    # Build model
    print("Building ResNet50 model...")
    model = build_resnet50(config.num_classes, device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Train
    trainer = Trainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=config,
        device=device,
        use_mixup=False,
    )

    print("Starting training...")
    trainer.train(optimizer, criterion, scheduler=scheduler)

    print(f"Training complete. Best val_loss: {trainer.best_val_loss:.4f}")
    print(f"Best model saved to {config.checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd D:/music_model_train && git add scripts/train_baseline.py && git commit -m "feat: add ResNet50 baseline training script"
```

---

### Task 10: Evaluation Script

**Files:**
- Create: `D:/music_model_train/scripts/evaluate.py`

- [ ] **Step 1: Write evaluate.py**

Write `D:/music_model_train/scripts/evaluate.py`:

```python
"""Evaluate trained models and compare AST vs ResNet50."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.training.config import TrainConfig, BaselineConfig, GENRE_NAMES
from src.data.dataset import get_dataloaders
from src.models.ast_model import build_ast_model
from src.models.resnet_baseline import build_resnet50
from src.evaluation.metrics import evaluate_model, print_eval_report, plot_confusion_matrix
from src.evaluation.compare import compare_models


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ast_cfg = TrainConfig()
    baseline_cfg = BaselineConfig()

    # Load data
    print("Loading dataset...")
    _, _, test_data = get_dataloaders()

    # Evaluate AST
    ast_ckpt = ast_cfg.checkpoint_dir / "best_model.pt"
    print(f"\n{'='*60}")
    print("Evaluating AST")
    print(f"{'='*60}")
    ast_model = build_ast_model(ast_cfg.model_name, ast_cfg.num_classes, device)
    ast_model.load_state_dict(torch.load(ast_ckpt, map_location=device, weights_only=True))
    ast_results = evaluate_model(ast_model, test_data, ast_cfg, device, GENRE_NAMES)
    print_eval_report(ast_results, GENRE_NAMES)

    # Evaluate ResNet50
    baseline_ckpt = baseline_cfg.checkpoint_dir / "best_model.pt"
    print(f"\n{'='*60}")
    print("Evaluating ResNet50 Baseline")
    print(f"{'='*60}")
    resnet_model = build_resnet50(baseline_cfg.num_classes, device)
    resnet_model.load_state_dict(torch.load(baseline_ckpt, map_location=device, weights_only=True))
    resnet_results = evaluate_model(resnet_model, test_data, baseline_cfg, device, GENRE_NAMES)
    print_eval_report(resnet_results, GENRE_NAMES)

    # Side-by-side comparison
    print(f"\n{'='*60}")
    print("HEAD-TO-HEAD COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'AST':>12} {'ResNet50':>12}")
    print("-" * 48)
    for metric in ["accuracy", "macro_f1", "weighted_f1"]:
        print(f"{metric:<20} {ast_results[metric]:>12.4f} {resnet_results[metric]:>12.4f}")

    print(f"\nConfusion matrices saved to ./logs/")

    # Save confusion matrices
    plot_confusion_matrix(
        ast_results["confusion_matrix"], GENRE_NAMES,
        title="AST - Confusion Matrix",
        save_path="./logs/ast_confusion_matrix.png"
    )
    plot_confusion_matrix(
        resnet_results["confusion_matrix"], GENRE_NAMES,
        title="ResNet50 - Confusion Matrix",
        save_path="./logs/resnet_confusion_matrix.png"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd D:/music_model_train && git add scripts/evaluate.py && git commit -m "feat: add evaluation and comparison script"
```

---

### Task 11: Inference Pipeline

**Files:**
- Create: `D:/music_model_train/src/inference/predictor.py`

- [ ] **Step 1: Write predictor.py**

Write `D:/music_model_train/src/inference/predictor.py`:

```python
"""Local inference: mp3 file -> mel spectrogram -> genre prediction."""
import torch
import numpy as np
import librosa
from PIL import Image
from torchvision import transforms

from src.training.config import GENRE_NAMES
from src.models.ast_model import build_ast_model


def _get_logits(output):
    return output.logits if hasattr(output, "logits") else output


class GenrePredictor:
    """Load a trained AST model and predict genre from mp3 files."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = build_ast_model(
            "MIT/ast-finetuned-audioset-10-epochs-0.4593",
            num_classes=16,
            device=self.device,
        )
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def audio_to_mel(self, audio_path: str) -> np.ndarray:
        """Extract mel spectrogram from mp3 file.

        Uses librosa with params that produce a visually similar spectrogram
        to the training data.
        """
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmax=11025
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Min-max normalize to [0, 1]
        mel_min = mel_db.min()
        mel_max = mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)
        # Convert to 0-255 uint8 image (mimicking JPEG spectrogram format)
        mel_img = (mel_norm * 255).astype(np.uint8)
        return mel_img

    def predict_segment(self, audio_segment: np.ndarray, sr: int = 22050) -> tuple:
        """Predict genre from an 11.4s audio segment.

        Returns (genre_name, confidence, top3_list).
        """
        # Extract mel
        mel_spec = librosa.feature.melspectrogram(
            y=audio_segment, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmax=11025
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_min, mel_max = mel_db.min(), mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)
        mel_img = Image.fromarray((mel_norm * 255).astype(np.uint8))

        # Transform and predict
        x = self.transform(mel_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = _get_logits(self.model(x))
            probs = torch.softmax(logits, dim=1)[0]

        top3_idx = probs.argsort(descending=True)[:3].cpu().numpy()
        top3 = [(GENRE_NAMES[i], probs[i].item()) for i in top3_idx]

        return GENRE_NAMES[top3_idx[0]], probs[top3_idx[0]].item(), top3

    def predict(self, audio_path: str, top_k: int = 3) -> list:
        """Predict genre for a full mp3 file.

        Slides an 11.4s window with 50% overlap, averages logits.
        Returns top-K predictions with confidence scores.
        """
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        window_samples = int(11.4 * sr)  # ~251k samples
        hop_samples = window_samples // 2   # 50% overlap

        all_logits = []
        for start in range(0, len(y) - window_samples + 1, hop_samples):
            segment = y[start : start + window_samples]
            mel_spec = librosa.feature.melspectrogram(
                y=segment, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmax=11025
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_min, mel_max = mel_db.min(), mel_db.max()
            mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)
            mel_img = Image.fromarray((mel_norm * 255).astype(np.uint8))

            x = self.transform(mel_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = _get_logits(self.model(x)).cpu()
            all_logits.append(logits)

        if not all_logits:
            # Audio too short, pad
            return self._predict_short(y, sr, top_k)

        # Average logits across all windows
        avg_logits = torch.stack(all_logits).mean(dim=0)[0]
        probs = torch.softmax(avg_logits, dim=0)

        top_k_idx = probs.argsort(descending=True)[:top_k].cpu().numpy()
        return [(GENRE_NAMES[i], round(probs[i].item(), 4)) for i in top_k_idx]

    def _predict_short(self, y: np.ndarray, sr: int, top_k: int) -> list:
        """Handle audio shorter than 11.4s by zero-padding."""
        target_len = int(11.4 * sr)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        return self.predict_segment(y, sr)[2]


def predict(audio_path: str, checkpoint_path: str = "./checkpoints/best_model.pt",
            device: str = "cpu") -> list:
    """Convenience function for quick inference."""
    predictor = GenrePredictor(checkpoint_path, device)
    return predictor.predict(audio_path)
```

- [ ] **Step 2: Commit**

```bash
cd D:/music_model_train && git add src/inference/predictor.py && git commit -m "feat: add local mp3 inference pipeline"
```

---

### Task 12: README

**Files:**
- Create: `D:/music_model_train/README.md`

- [ ] **Step 1: Write README.md**

Write `D:/music_model_train/README.md`:

```markdown
# Music Genre Classification

16-class fine-grained music genre classifier using AST (Audio Spectrogram Transformer) fine-tuned on the CCMUSIC dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Training (on AutoDL or any GPU machine)

```bash
# ResNet50 baseline (~2-3 hours)
python scripts/train_baseline.py

# AST fine-tuning (~4-6 hours)
python scripts/train_ast.py
```

## Evaluation

```bash
python scripts/evaluate.py
```

Outputs accuracy, macro/weighted F1, per-class report, confusion matrices (AST vs ResNet50).

## Local Inference

```python
from src.inference.predictor import predict

results = predict("path/to/song.mp3", checkpoint_path="./checkpoints/best_model.pt")
for genre, confidence in results:
    print(f"{genre}: {confidence:.2%}")
```

## Project Structure

- `src/data/` - Dataset loading and transforms (SpecAugment, Mixup)
- `src/models/` - AST model builder and ResNet50 baseline
- `src/training/` - Trainer with early stopping and config
- `src/evaluation/` - Metrics, confusion matrix, model comparison
- `src/inference/` - mp3 to genre prediction pipeline
- `scripts/` - Training and evaluation entry points

## Dataset

[CCMUSIC Music Genre Dataset](https://huggingface.co/datasets/ccmusic-database/music_genre) - 36,375 mel spectrogram segments, 16 fine-grained genres, MIT license.
```

- [ ] **Step 2: Final commit**

```bash
cd D:/music_model_train && git add README.md && git commit -m "docs: add README with usage instructions"
```

---

### Task 13: Smoke Test

**Files:**
- Create: `D:/music_model_train/scripts/smoke_test.py`

- [ ] **Step 1: Write smoke_test.py**

Write `D:/music_model_train/scripts/smoke_test.py`:

```python
"""Smoke test: verify all modules import and data pipeline works on CPU with small subset."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.training.config import TrainConfig, BaselineConfig, GENRE_NAMES
from src.data.dataset import get_dataloaders, collate_fn
from src.data.transforms import (
    get_train_transforms, get_eval_transforms, SpecAugment, mixup_data, mixup_loss
)
from src.models.ast_model import build_ast_model, get_parameter_groups
from src.models.resnet_baseline import build_resnet50
from src.evaluation.metrics import compute_mid_level_accuracy
from PIL import Image
import numpy as np


def test_config():
    """Verify configs instantiate correctly."""
    cfg = TrainConfig()
    assert cfg.num_classes == 16
    assert cfg.batch_size == 32
    bcfg = BaselineConfig()
    assert bcfg.num_classes == 16
    assert len(GENRE_NAMES) == 16
    print("[PASS] Config")


def test_transforms():
    """Verify transforms produce correct tensor shapes."""
    train_tf = get_train_transforms(224)
    eval_tf = get_eval_transforms(224)
    # Create a fake mel spectrogram image (random pixels)
    fake_img = Image.fromarray(np.random.randint(0, 255, (128, 500), dtype=np.uint8))
    train_out = train_tf(fake_img)
    assert train_out.shape == (1, 224, 224)
    eval_out = eval_tf(fake_img)
    assert eval_out.shape == (1, 224, 224)
    print("[PASS] Transforms")


def test_specaugment():
    """Verify SpecAugment runs without error and masks regions."""
    specaug = SpecAugment(time_mask_param=10, freq_mask_param=2)
    x = torch.randn(4, 1, 224, 224)
    x_aug = specaug(x.clone())
    assert x_aug.shape == x.shape
    print("[PASS] SpecAugment")


def test_mixup():
    """Verify Mixup blending."""
    x = torch.randn(4, 1, 224, 224)
    y = torch.randint(0, 16, (4,))
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.5)
    assert mixed_x.shape == x.shape
    assert 0 <= lam <= 1
    print("[PASS] Mixup")


def test_mid_level_accuracy():
    """Verify mid-level rollup works."""
    preds = np.array([0, 5, 12, 8])
    labels = np.array([0, 12, 12, 8])
    acc = compute_mid_level_accuracy(preds, labels)
    # pred 0->0 (Classic) vs label 0->0 = correct
    # pred 5->5 (Pop) vs label 12->8 (Rock) = wrong
    # pred 12->8 (Rock) vs label 12->8 (Rock) = correct
    # pred 8->6 (Dance) vs label 8->6 (Dance) = correct
    # 3/4 correct
    assert acc == 0.75, f"Expected 0.75, got {acc}"
    print("[PASS] Mid-level accuracy")


def test_collate():
    """Verify collate_fn with mock PIL images."""
    mock_batch = [
        {"mel": Image.fromarray(np.random.randint(0, 255, (128, 500), dtype=np.uint8)),
         "thr_level_label": 5},
        {"mel": Image.fromarray(np.random.randint(0, 255, (128, 500), dtype=np.uint8)),
         "thr_level_label": 12},
    ]
    images, labels = collate_fn(mock_batch)
    assert len(images) == 2
    assert labels.tolist() == [5, 12]
    print("[PASS] Collate function")


def test_genre_names():
    """Verify genre names are correct."""
    assert GENRE_NAMES[0] == "Symphony"
    assert GENRE_NAMES[4] == "Soul_or_r_and_b"
    assert GENRE_NAMES[15] == "Acoustic_pop"
    print("[PASS] Genre names")


if __name__ == "__main__":
    print("Running smoke tests...\n")
    test_config()
    test_transforms()
    test_specaugment()
    test_mixup()
    test_mid_level_accuracy()
    test_collate()
    test_genre_names()
    print("\n=== All smoke tests passed! ===")
```

- [ ] **Step 2: Run smoke test**

```bash
cd D:/music_model_train && python scripts/smoke_test.py
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
cd D:/music_model_train && git add scripts/smoke_test.py && git commit -m "test: add smoke tests for all modules"
```
