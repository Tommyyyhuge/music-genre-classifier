# Music Genre Classification

16-class fine-grained music genre classifier using AST (Audio Spectrogram Transformer) fine-tuned on the CCMUSIC dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Training (on AutoDL or any GPU machine)

```bash
# ResNet50 baseline (~2-3 hours)
python scripts/train_baseline.py

# AST fine-tuning (~4-6 hours)
python scripts/train_ast.py
```

## Evaluation

```bash
python scripts/evaluate.py
```

Outputs accuracy, macro/weighted F1, per-class report, confusion matrices (AST vs ResNet50).

## Local Inference

```python
from src.inference.predictor import predict

results = predict("path/to/song.mp3", checkpoint_path="./checkpoints/best_model.pt")
for genre, confidence in results:
    print(f"{genre}: {confidence:.2%}")
```

## Project Structure

- `src/data/` - Dataset loading and transforms (SpecAugment, Mixup)
- `src/models/` - AST model builder and ResNet50 baseline
- `src/training/` - Trainer with early stopping and config
- `src/evaluation/` - Metrics, confusion matrix, model comparison
- `src/inference/` - mp3 to genre prediction pipeline
- `scripts/` - Training and evaluation entry points

## Dataset

[CCMUSIC Music Genre Dataset](https://huggingface.co/datasets/ccmusic-database/music_genre) - 36,375 mel spectrogram segments, 16 fine-grained genres, MIT license.
