"""Smoke test: verify all modules import and work correctly on CPU with mock data."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

from src.training.config import TrainConfig, BaselineConfig, GENRE_NAMES
from src.data.dataset import load_splits, collate_fn
from src.data.transforms import (
    get_train_transforms, get_eval_transforms, SpecAugment, mixup_data, mixup_loss
)
from src.evaluation.metrics import compute_mid_level_accuracy


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
    fake_img = Image.fromarray(np.random.randint(0, 255, (128, 500), dtype=np.uint8))
    train_out = train_tf(fake_img)
    assert train_out.shape == (1, 224, 224)
    eval_out = eval_tf(fake_img)
    assert eval_out.shape == (1, 224, 224)
    print("[PASS] Transforms")


def test_specaugment():
    """Verify SpecAugment runs without error and produces correct shape."""
    specaug = SpecAugment(time_mask_param=10, freq_mask_param=2)
    x = torch.randn(4, 1, 224, 224)
    x_aug = specaug(x.clone())
    assert x_aug.shape == x.shape
    print("[PASS] SpecAugment")


def test_mixup():
    """Verify Mixup blending produces correct shapes."""
    x = torch.randn(4, 1, 224, 224)
    y = torch.randint(0, 16, (4,))
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.5)
    assert mixed_x.shape == x.shape
    assert 0 <= lam <= 1
    print("[PASS] Mixup")


def test_mid_level_accuracy():
    """Verify mid-level rollup works correctly."""
    preds = np.array([0, 5, 12, 8])
    labels = np.array([0, 12, 12, 8])
    acc = compute_mid_level_accuracy(preds, labels)
    # pred 0->0 (Classic) vs label 0->0 (Classic) = correct
    # pred 5->2 (Pop) vs label 12->5 (Rock) = wrong
    # pred 12->5 (Rock) vs label 12->5 (Rock) = correct
    # pred 8->3 (Dance) vs label 8->3 (Dance) = correct
    # 3/4 = 0.75
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
