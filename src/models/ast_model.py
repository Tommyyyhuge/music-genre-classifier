import torch
from transformers import ASTForAudioClassification


def build_ast_model(model_name: str, num_classes: int, device: torch.device):
    """Load AST model with AudioSet pretrained weights, replace classifier head.

    Args:
        model_name: HuggingFace model ID (e.g. "MIT/ast-finetuned-audioset-10-10-0.4593")
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
