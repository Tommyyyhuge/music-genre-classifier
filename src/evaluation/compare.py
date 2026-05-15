"""AST vs ResNet50 comparison utilities."""
from src.evaluation.metrics import evaluate_model, compute_mid_level_accuracy, plot_confusion_matrix


def compare_models(ast_model, resnet_model, test_data, ast_cfg, baseline_cfg, device, label_names):
    """Evaluate both models side-by-side, return comparison dict."""
    print("\n>>> Evaluating AST model...")
    ast_results = evaluate_model(ast_model, test_data, ast_cfg, device, label_names)

    print("\n>>> Evaluating ResNet50 baseline...")
    resnet_results = evaluate_model(resnet_model, test_data, baseline_cfg, device, label_names)

    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'AST':>12} {'ResNet50':>12}")
    print("-" * 48)
    for metric in ["accuracy", "macro_f1", "weighted_f1"]:
        print(f"{metric:<20} {ast_results[metric]:>12.4f} {resnet_results[metric]:>12.4f}")

    mid_ast = compute_mid_level_accuracy(ast_results["all_preds"], ast_results["all_labels"])
    mid_resnet = compute_mid_level_accuracy(resnet_results["all_preds"], resnet_results["all_labels"])
    print(f"{'Mid-level Acc':<20} {mid_ast:>12.4f} {mid_resnet:>12.4f}")

    plot_confusion_matrix(
        ast_results["confusion_matrix"], label_names,
        title="AST Confusion Matrix", save_path="./logs/ast_confusion_matrix.png"
    )
    plot_confusion_matrix(
        resnet_results["confusion_matrix"], label_names,
        title="ResNet50 Confusion Matrix", save_path="./logs/resnet_confusion_matrix.png"
    )

    return {
        "ast": ast_results,
        "resnet": resnet_results,
    }
