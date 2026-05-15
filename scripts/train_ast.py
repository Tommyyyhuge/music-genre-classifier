"""Train AST model for 16-class music genre classification."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.training.config import TrainConfig
from src.data.dataset import load_splits
from src.models.ast_model import build_ast_model, get_parameter_groups
from src.training.trainer import Trainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = TrainConfig()

    print("Loading dataset...")
    train_data, val_data, test_data = load_splits()

    print("Building AST model...")
    model = build_ast_model(config.model_name, config.num_classes, device)

    param_groups = get_parameter_groups(
        model, config.lr_backbone, config.lr_head, config.weight_decay
    )
    optimizer = AdamW(param_groups)

    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs - config.warmup_epochs)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    trainer = Trainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=config,
        device=device,
        use_mixup=config.use_mixup,
    )

    print("Starting training...")
    trainer.train(optimizer, criterion, scheduler=scheduler)

    print(f"Training complete. Best val_loss: {trainer.best_val_loss:.4f}")
    print(f"Best model saved to {config.checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
