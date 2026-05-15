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

from src.data.transforms import get_eval_transforms
from src.data.dataset import collate_fn


@torch.no_grad()
def evaluate_model(model, test_data, config, device, label_names=None):
    """Run full evaluation on test set.

    Returns dict with keys: accuracy, macro_f1, weighted_f1, per_class,
    confusion_matrix, all_preds, all_labels.
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
    """Roll up 16 fine-grained predictions to 9 mid-level classes.

    Mid-level mapping:
    0-3: Classic family -> mid 0
    4: Soul/R&B -> mid 1
    5-7: Pop -> mid 2
    8-9: Dance -> mid 3
    10-11: Indie -> mid 4
    12-15: Rock -> mid 5
    """
    fine_to_mid = {
        0: 0, 1: 0, 2: 0, 3: 0,   # Classic family
        4: 1,                       # Soul/R&B
        5: 2, 6: 2, 7: 2,          # Pop
        8: 3, 9: 3,                # Dance
        10: 4, 11: 4,              # Indie
        12: 5, 13: 5, 14: 5, 15: 5,  # Rock
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
    print("EVALUATION REPORT")
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

    mid_acc = compute_mid_level_accuracy(results["all_preds"], results["all_labels"])
    print(f"\nMid-level (6-group rollup) Accuracy: {mid_acc:.4f}")
    print(f"{'='*60}\n")
