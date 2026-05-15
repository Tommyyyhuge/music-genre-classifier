import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
import copy

from src.data.transforms import (
    get_train_transforms,
    get_eval_transforms,
    SpecAugment,
    mixup_data,
    mixup_loss,
    collate_fn,
)


def _get_logits(output):
    """Extract logits tensor from model output.

    HF transformers models return objects with .logits attribute;
    timm models return tensors directly.
    """
    return output.logits if hasattr(output, "logits") else output


class Trainer:
    """Generic trainer for music genre classification models."""

    def __init__(
        self,
        model: nn.Module,
        train_data,
        val_data,
        config,
        device: torch.device,
        use_mixup: bool = True,
    ):
        self.model = model
        self.device = device
        self.config = config
        self.use_mixup = use_mixup

        self.train_transform = get_train_transforms(config.image_size)
        self.eval_transform = get_eval_transforms(config.image_size)
        self.specaug = (
            SpecAugment(config.time_mask_param, config.freq_mask_param)
            if getattr(config, "use_specaugment", False)
            else None
        )

        self.train_data = train_data
        self.val_data = val_data

        self.writer = SummaryWriter(log_dir=str(config.log_dir))
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _transform_batch(images, transform):
        """Apply transform to a list of PIL images, return tensor (B, 1, H, W)."""
        batch = torch.stack([transform(img) for img in images])
        return batch

    def _train_epoch(self, optimizer, criterion, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        indices = np.random.permutation(len(self.train_data))
        batch_size = self.config.batch_size

        pbar = tqdm(range(0, len(indices), batch_size), desc=f"Train Epoch {epoch}")
        for start in pbar:
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]
            batch_items = [self.train_data[int(i)] for i in batch_indices]
            images, labels = collate_fn(batch_items)

            x = self._transform_batch(images, self.train_transform)
            labels = labels.to(self.device)
            x = x.to(self.device)

            # SpecAugment
            if self.specaug is not None:
                x = self.specaug(x)

            # Mixup
            if self.use_mixup:
                x, y_a, y_b, lam = mixup_data(x, labels, alpha=self.config.mixup_alpha)
                output = self.model(x)
                logits = _get_logits(output)
                loss = mixup_loss(criterion, logits, y_a, y_b, lam)
                preds = logits.argmax(dim=1)
                total_correct += (
                    lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()
                ).sum().item()
            else:
                output = self.model(x)
                logits = _get_logits(output)
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()

            total_samples += len(labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{total_correct / total_samples:.4f}",
            })

        if total_samples == 0:
            return 0.0, 0.0
        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def _validate_epoch(self, criterion, epoch):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        batch_size = self.config.batch_size
        pbar = tqdm(range(0, len(self.val_data), batch_size), desc=f"Val Epoch {epoch}")
        for start in pbar:
            end = min(start + batch_size, len(self.val_data))
            batch_items = [self.val_data[int(i)] for i in range(start, end)]
            images, labels = collate_fn(batch_items)

            x = self._transform_batch(images, self.eval_transform)
            x = x.to(self.device)
            labels = labels.to(self.device)

            output = self.model(x)
            logits = _get_logits(output)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * len(labels)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

        if total_samples == 0:
            return 0.0, 0.0
        val_loss = total_loss / total_samples
        val_acc = total_correct / total_samples
        return val_loss, val_acc

    def train(self, optimizer, criterion, scheduler=None):
        patience = self.config.early_stop_patience

        # Reset early-stopping state for fresh training run
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_no_improve = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self._train_epoch(optimizer, criterion, epoch)
            val_loss, val_acc = self._validate_epoch(criterion, epoch)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Acc/train", train_acc, epoch)
            self.writer.add_scalar("Acc/val", val_acc, epoch)
            self.writer.flush()

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

            # Early stopping + checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
                torch.save(
                    self.best_model_state,
                    self.config.checkpoint_dir / "best_model.pt",
                )
                print(f"  -> Best model saved (val_loss={val_loss:.4f})")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.writer.close()
        # Restore best weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
