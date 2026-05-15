"""Train ResNet50 baseline for 16-class music genre classification."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.training.config import BaselineConfig
from src.data.dataset import load_splits
from src.models.resnet_baseline import build_resnet50
from src.training.trainer import Trainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = BaselineConfig()

    print("Loading dataset...")
    train_data, val_data, test_data = load_splits()

    print("Building ResNet50 model...")
    model = build_resnet50(config.num_classes, device)

    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=config,
        device=device,
        use_mixup=False,
    )

    print("Starting training...")
    trainer.train(optimizer, criterion, scheduler=scheduler)

    print(f"Training complete. Best val_loss: {trainer.best_val_loss:.4f}")
    print(f"Best model saved to {config.checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
