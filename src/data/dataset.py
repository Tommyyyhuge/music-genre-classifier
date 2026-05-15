import torch
from PIL import Image
from datasets import load_dataset


def load_splits():
    """Load CCMUSIC dataset and return train/val/test splits as HF Dataset objects.

    Returns raw HuggingFace Dataset objects (not PyTorch DataLoaders).
    The Trainer handles batching manually to support per-batch Mixup.
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
    images = []
    labels = []
    for item in batch:
        mel_img = item["mel"]
        if isinstance(mel_img, Image.Image):
            mel_img = mel_img.convert("RGB")
        images.append(mel_img)
        labels.append(item["thr_level_label"])

    return images, torch.tensor(labels, dtype=torch.long)
