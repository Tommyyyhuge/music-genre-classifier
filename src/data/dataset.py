import io
import torch
from PIL import Image


def load_splits():
    """Load CCMUSIC dataset from ModelScope (or fallback HuggingFace).

    Returns sequence-like objects supporting __len__ and integer indexing.
    The Trainer handles batching manually to support per-batch Mixup.
    """
    try:
        from modelscope.msdatasets import MsDataset
        print("Loading from ModelScope...")
        train_data = MsDataset.load(
            "ccmusic-database/music_genre", split="train", trust_remote_code=True
        )
        val_data = MsDataset.load(
            "ccmusic-database/music_genre", split="validation", trust_remote_code=True
        )
        test_data = MsDataset.load(
            "ccmusic-database/music_genre", split="test", trust_remote_code=True
        )
        print(
            f"Loaded: train={len(train_data)}, "
            f"val={len(val_data)}, test={len(test_data)}"
        )
    except Exception as e:
        print(f"ModelScope failed ({e}), falling back to HuggingFace...")
        from datasets import load_dataset
        dataset = load_dataset("ccmusic-database/music_genre")
        train_data = dataset["train"]
        val_data = dataset["validation"]
        test_data = dataset["test"]

    return train_data, val_data, test_data


def _to_pil(image_value):
    """Convert various image formats to PIL Image."""
    if isinstance(image_value, Image.Image):
        return image_value
    if isinstance(image_value, bytes):
        return Image.open(io.BytesIO(image_value))
    if isinstance(image_value, dict):
        raw = image_value.get("bytes") or image_value.get("path")
        if isinstance(raw, bytes):
            return Image.open(io.BytesIO(raw))
        if isinstance(raw, str):
            return Image.open(raw)
    raise TypeError(f"Cannot convert {type(image_value)} to PIL Image")


def collate_fn(batch):
    """Custom collate: extract mel images and thr_level_label from samples."""
    images = []
    labels = []
    for item in batch:
        mel_img = _to_pil(item["mel"]).convert("RGB")
        images.append(mel_img)
        labels.append(item["thr_level_label"])

    return images, torch.tensor(labels, dtype=torch.long)
