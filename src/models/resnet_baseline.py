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
