import torch
from torchvision import transforms
import numpy as np


def _base_transforms(image_size: int = 224):
    """Shared transforms: grayscale, resize, to tensor, normalize to [-1, 1]."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_train_transforms(image_size: int = 224):
    """Transforms for training."""
    return _base_transforms(image_size)


def get_eval_transforms(image_size: int = 224):
    """Transforms for validation/test."""
    return _base_transforms(image_size)


class SpecAugment:
    """Frequency and time masking for spectrogram augmentation.

    Args:
        num_freq_masks: Number of frequency bands to mask.
        time_mask_pct: Percentage of time dimension to mask.
    """

    def __init__(self, time_mask_param: int = 10, freq_mask_param: int = 2):
        self.num_freq_masks = freq_mask_param
        self.time_mask_pct = time_mask_param

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, H, W)
        x = x.clone()
        B, C, H, W = x.shape
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = np.random.randint(1, max(2, H // 10 + 1))
            f0 = np.random.randint(0, H - f)
            x[:, :, f0 : f0 + f, :] = 0.0
        # Time masking
        t = int(W * self.time_mask_pct / 100)
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
