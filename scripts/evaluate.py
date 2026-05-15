"""Evaluate trained models and compare AST vs ResNet50."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.training.config import TrainConfig, BaselineConfig, GENRE_NAMES
from src.data.dataset import load_splits
from src.models.ast_model import build_ast_model
from src.models.resnet_baseline import build_resnet50
from src.evaluation.metrics import evaluate_model, print_eval_report, plot_confusion_matrix


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ast_cfg = TrainConfig()
    baseline_cfg = BaselineConfig()

    print("Loading dataset...")
    _, _, test_data = load_splits()

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

    print(f"\nConfusion matrices saved to ./logs/")


if __name__ == "__main__":
    main()
