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
