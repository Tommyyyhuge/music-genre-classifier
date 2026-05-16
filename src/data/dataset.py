import io
import os
import glob
import torch
from PIL import Image
from datasets import Dataset, load_dataset


def load_splits():
    """Load CCMUSIC dataset from ModelScope cache or fallback HuggingFace.

    Returns sequence-like objects supporting __len__ and integer indexing.
    The Trainer handles batching manually to support per-batch Mixup.
    """
    # Step 1: trigger ModelScope download if not cached
    try:
        from modelscope.msdatasets import MsDataset
        print("Checking ModelScope cache...")
        MsDataset.load(
            "ccmusic-database/music_genre", split="train", trust_remote_code=True
        )
    except Exception:
        pass  # download may succeed even if as_dataset() fails

    # Step 2: find arrow files in ModelScope cache
    cache_base = os.path.expanduser("~/.cache/modelscope/hub/datasets")
    pattern = os.path.join(
        cache_base,
        "ccmusic-database___music_genre", "**", "music_genre-train.arrow",
    )
    matches = glob.glob(pattern, recursive=True)

    if matches:
        data_dir = os.path.dirname(matches[0])
        print(f"Loading from ModelScope cache: {data_dir}")
        import pyarrow as pa

        def _read_arrow(name):
            path = os.path.join(data_dir, f"music_genre-{name}.arrow")
            with pa.ipc.open_file(path) as f:
                return Dataset(f.read_all())

        train_data = _read_arrow("train")
        val_data = _read_arrow("validation")
        test_data = _read_arrow("test")
    else:
        print("Falling back to HuggingFace...")
        dataset = load_dataset("ccmusic-database/music_genre")
        train_data = dataset["train"]
        val_data = dataset["validation"]
        test_data = dataset["test"]

    print(
        f"Loaded: train={len(train_data)}, "
        f"val={len(val_data)}, test={len(test_data)}"
    )
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
