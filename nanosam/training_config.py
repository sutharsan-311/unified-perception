# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training configuration for OWL-SAM adapter."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for adapter training."""

    # Data
    dataset_name: str = "coco"
    dataset_split: str = "train2017"
    num_images: int = 5000
    image_size: int = 768
    batch_size: int = 4
    num_workers: int = 4

    # Model
    owl_model_name: str = "google/owlvit-base-patch32"
    owl_pretrained: bool = True
    adapter_hidden_dim: int = 768
    adapter_mlp_hidden_dim: int = 512
    adapter_output_dim: int = 256

    # Training
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    gradient_clip: float = 1.0

    # Loss
    loss_type: str = "combined"  # "mse", "cosine", or "combined"
    mse_weight: float = 0.5
    cosine_weight: float = 0.5

    # Optimization
    optimizer: str = "adam"
    scheduler: str = "cosine"
    save_interval: int = 100  # Save every N batches

    # Validation
    validation_interval: int = 500  # Validate every N batches
    validation_size: int = 100  # Number of images for validation

    # Checkpoint
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_name: str = "adapter_phase_c"

    # Device
    device: str = "cuda"
    fp16: bool = False
    compile_model: bool = False

    # Logging
    log_interval: int = 10
    log_dir: Path = Path("logs")
    wandb_enabled: bool = False
    wandb_project: str = "unified-perception"

    def __post_init__(self):
        """Create directories if needed."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# Presets for different scenarios

COCO_SMALL = TrainingConfig(
    num_images=1000,
    num_epochs=5,
    batch_size=2,
)

COCO_MEDIUM = TrainingConfig(
    num_images=5000,
    num_epochs=10,
    batch_size=4,
)

COCO_LARGE = TrainingConfig(
    num_images=50000,
    num_epochs=20,
    batch_size=8,
)

HOSPITAL_FINETUNE = TrainingConfig(
    num_images=500,
    num_epochs=5,
    batch_size=2,
    learning_rate=5e-4,  # Lower LR for fine-tuning
    warmup_steps=100,
)
