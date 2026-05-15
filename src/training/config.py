from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    """AST fine-tuning configuration."""
    model_name: str = "MIT/ast-finetuned-audioset-10-epochs-0.4593"
    num_classes: int = 16
    image_size: int = 224

    batch_size: int = 32
    epochs: int = 30
    lr_backbone: float = 5e-5
    lr_head: float = 5e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    warmup_epochs: int = 5
    early_stop_patience: int = 5

    num_workers: int = 4

    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")

    use_specaugment: bool = True
    time_mask_param: int = 10
    freq_mask_param: int = 2
    use_mixup: bool = True
    mixup_alpha: float = 0.5


@dataclass(frozen=True)
class BaselineConfig:
    """ResNet50 baseline configuration."""
    model_name: str = "resnet50"
    num_classes: int = 16
    image_size: int = 224

    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 5

    num_workers: int = 4

    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")

    use_specaugment: bool = True
    time_mask_param: int = 10
    freq_mask_param: int = 2


# Genre label names (Level 3, 16 classes, 0-indexed order)
GENRE_NAMES: tuple[str, ...] = (
    "Symphony",
    "Opera",
    "Solo",
    "Chamber",
    "Soul_or_r_and_b",
    "Pop_vocal_ballad",
    "Adult_contemporary",
    "Teen_pop",
    "Contemporary_dance_pop",
    "Dance_pop",
    "Classic_indie_pop",
    "Chamber_cabaret_and_art_pop",
    "Adult_alternative_rock",
    "Uplifting_anthemic_rock",
    "Soft_rock",
    "Acoustic_pop",
)
