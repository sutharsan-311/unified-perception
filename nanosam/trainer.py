# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training loop for OWL-SAM adapter."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from datetime import datetime
import time


class AdapterLoss(nn.Module):
    """Combined loss function for adapter training."""

    def __init__(
        self,
        mse_weight: float = 0.5,
        cosine_weight: float = 0.5,
    ):
        """
        Args:
            mse_weight: Weight for MSE loss
            cosine_weight: Weight for cosine similarity loss
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight

    def forward(
        self,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            pred_features: Predicted features from adapter (B, C, H, W)
            target_features: Target SAM features (B, C, H, W)

        Returns:
            loss: Scalar loss value
        """
        # MSE Loss - pixel-level feature matching
        mse_loss = F.mse_loss(pred_features, target_features)

        # Cosine Similarity Loss - direction matching
        # Reshape to (B*H*W, C) for cosine similarity
        B, C, H, W = pred_features.shape
        pred_flat = pred_features.permute(0, 2, 3, 1).reshape(B * H * W, C)
        target_flat = target_features.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Cosine similarity: 1 - similarity (want to maximize similarity)
        cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1)
        cosine_loss = 1 - cosine_sim.mean()

        # Combined loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.cosine_weight * cosine_loss
        )

        return total_loss


class AdapterTrainer:
    """Trainer for OWL-SAM adapter."""

    def __init__(
        self,
        adapter: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config=None,
        owl_encoder=None,
        sam_encoder=None,
    ):
        """
        Args:
            adapter: OwlToSamAdapter module
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: TrainingConfig object
        """
        self.adapter = adapter
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.device if config else "cuda")
        self.owl_encoder = owl_encoder
        self.sam_encoder = sam_encoder
        self.phase_c = owl_encoder is not None and sam_encoder is not None

        # Move adapter to device
        self.adapter = self.adapter.to(self.device)

        # Loss function
        self.criterion = AdapterLoss(
            mse_weight=config.mse_weight if config else 0.5,
            cosine_weight=config.cosine_weight if config else 0.5,
        )

        # Optimizer
        self.optimizer = Adam(
            self.adapter.parameters(),
            lr=config.learning_rate if config else 1e-3,
            weight_decay=config.weight_decay if config else 1e-5,
        )

        # Learning rate scheduler
        if config and config.scheduler == "cosine":
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=config.warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=len(train_loader) * (config.num_epochs - 1),
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[config.warmup_steps],
            )
        else:
            self.scheduler = None

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.start_time = None

        # Checkpoint directory
        if config:
            self.checkpoint_dir = config.checkpoint_dir
            self.checkpoint_name = config.checkpoint_name
        else:
            self.checkpoint_dir = Path("checkpoints")
            self.checkpoint_name = "adapter"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with epoch metrics
        """
        self.adapter.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (images, _) in enumerate(self.train_loader):
            images = images.to(self.device)

            self.optimizer.zero_grad()

            if self.phase_c:
                owl_features = self.owl_encoder(images)
                target_features = self.sam_encoder(images)
            else:
                owl_features = torch.randn(images.shape[0], 576, 768, device=self.device, dtype=images.dtype)
                target_features = torch.randn(images.shape[0], 256, 64, 64, device=self.device, dtype=images.dtype)

            pred_features = self.adapter(owl_features)

            # Compute loss
            loss = self.criterion(pred_features, target_features)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config and self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.adapter.parameters(),
                    self.config.gradient_clip
                )

            # Optimizer step
            self.optimizer.step()

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Metrics
            total_loss += loss.item()
            num_batches += 1

            # Periodic checkpoint — survives a Colab kernel restart mid-epoch.
            # Rolling file per epoch; resume later with --load-checkpoint.
            if (
                self.config and
                self.config.save_interval and
                batch_idx > 0 and
                batch_idx % self.config.save_interval == 0
            ):
                self.save_checkpoint(epoch=epoch)
                print(f"  → Periodic checkpoint at batch {batch_idx}")

            # Logging
            if self.config and batch_idx % self.config.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.6f} (Avg: {avg_loss:.6f}) "
                    f"LR: {lr:.2e}"
                )

            # Validation
            if (
                self.config and
                self.val_loader and
                batch_idx % self.config.validation_interval == 0 and
                batch_idx > 0
            ):
                val_loss = self.validate()
                print(f"  → Validation Loss: {val_loss:.6f}")

                # Save checkpoint if best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
                    print(f"  → Saved best checkpoint!")

        return {"train_loss": total_loss / max(num_batches, 1)}

    def validate(self) -> float:
        """
        Run validation.

        Returns:
            Average validation loss
        """
        self.adapter.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for images, _ in self.val_loader:
                images = images.to(self.device)

                if self.phase_c:
                    owl_features = self.owl_encoder(images)
                    target_features = self.sam_encoder(images)
                else:
                    owl_features = torch.randn(images.shape[0], 576, 768, device=self.device, dtype=images.dtype)
                    target_features = torch.randn(images.shape[0], 256, 64, 64, device=self.device, dtype=images.dtype)

                pred_features = self.adapter(owl_features)
                loss = self.criterion(pred_features, target_features)

                total_loss += loss.item()
                num_batches += 1

        self.adapter.train()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return total_loss / max(num_batches, 1)

    def train(self) -> Dict:
        """
        Train for full number of epochs.

        Returns:
            Training history
        """
        self.start_time = time.time()
        history = {"train_losses": [], "val_losses": []}

        num_epochs = self.config.num_epochs if self.config else 10

        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*70}")

            # Train
            epoch_metrics = self.train_epoch(epoch)
            history["train_losses"].append(epoch_metrics["train_loss"])
            self.train_losses.append(epoch_metrics["train_loss"])

            # Validate
            if self.val_loader:
                val_loss = self.validate()
                history["val_losses"].append(val_loss)
                self.val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.6f}")

                # Track and save the best model by validation loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
                    print(f"  → Saved best checkpoint!")

                # Periodic epoch snapshot
                if epoch % 5 == 0:
                    self.save_checkpoint(epoch=epoch)

        # Save final checkpoint
        self.save_checkpoint(epoch=num_epochs, is_final=True)

        # Training summary
        elapsed = time.time() - self.start_time
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Total time: {elapsed/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Checkpoint saved: {self.checkpoint_dir}")
        print(f"{'='*70}\n")

        return history

    def save_checkpoint(
        self,
        epoch: Optional[int] = None,
        is_best: bool = False,
        is_final: bool = False,
    ):
        """
        Save checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best checkpoint
            is_final: Whether this is the final checkpoint
        """
        if is_best:
            checkpoint_path = self.checkpoint_dir / f"{self.checkpoint_name}_best.pth"
        elif is_final:
            checkpoint_path = self.checkpoint_dir / f"{self.checkpoint_name}_final.pth"
        else:
            checkpoint_path = self.checkpoint_dir / f"{self.checkpoint_name}_epoch{epoch}.pth"

        checkpoint = {
            "adapter_state_dict": self.adapter.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.adapter.load_state_dict(checkpoint["adapter_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        print(f"Checkpoint loaded from {checkpoint_path}")
