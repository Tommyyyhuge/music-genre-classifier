import io
import os
import torch
from PIL import Image
from datasets import load_dataset


def load_splits():
    """Load CCMUSIC dataset from ModelScope cache or fallback HuggingFace.

    Returns sequence-like objects supporting __len__ and integer indexing.
    The Trainer handles batching manually to support per-batch Mixup.
    """
    # Step 1: ensure data is downloaded via ModelScope
    try:
        from modelscope.msdatasets import MsDataset
        print("Downloading from ModelScope (if not cached)...")
        MsDataset.load(
            "ccmusic-database/music_genre", split="train", trust_remote_code=True
        )
    except Exception as e:
        print(f"  (ModelScope load attempt: {e})")

    # Step 2: load from ModelScope cache using HF datasets
    cache_root = os.path.expanduser("~/.cache/modelscope/hub/datasets")
    script_dir = None
    if os.path.isdir(cache_root):
        for root, dirs, files in os.walk(cache_root):
            if "music_genre.py" in files and "ccmusic" in root:
                script_dir = root
                break

    if script_dir:
        print(f"Loading from ModelScope cache: {script_dir}")
        dataset = load_dataset(script_dir, trust_remote_code=True)
    else:
        print("Falling back to HuggingFace...")
        dataset = load_dataset("ccmusic-database/music_genre")

    print(
        f"Loaded: train={len(dataset['train'])}, "
        f"val={len(dataset['validation'])}, test={len(dataset['test'])}"
    )
    return dataset["train"], dataset["validation"], dataset["test"]


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
