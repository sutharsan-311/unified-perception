#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train OWL-SAM adapter on COCO dataset.

Usage:
    python train.py --dataset-dir data/coco --num-images 5000 --batch-size 4
"""

import argparse
import sys
from pathlib import Path

import torch

# Add nanosam to path
sys.path.insert(0, str(Path(__file__).parent))

from nanosam.adapter import OwlToSamAdapter
from nanosam.training_config import TrainingConfig, COCO_MEDIUM
from nanosam.data_loader import TrainingDataLoader
from nanosam.trainer import AdapterTrainer


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train OWL-SAM adapter")

    # Data arguments
    parser.add_argument("--dataset-dir", default="data/coco",
                        help="Root directory of COCO dataset")
    parser.add_argument("--num-images", type=int, default=5000,
                        help="Number of training images")
    parser.add_argument("--validation-size", type=int, default=100,
                        help="Number of validation images")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")

    # Loss arguments
    parser.add_argument("--mse-weight", type=float, default=0.5,
                        help="Weight for MSE loss")
    parser.add_argument("--cosine-weight", type=float, default=0.5,
                        help="Weight for cosine similarity loss")

    # Device arguments
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on (cuda or cpu)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 precision")

    # Checkpoint arguments
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--load-checkpoint", default=None,
                        help="Path to checkpoint to load")

    # Phase C arguments
    parser.add_argument("--phase-c", action="store_true",
                        help="Use real OWL + SAM encoders (Phase C)")
    parser.add_argument("--sam-checkpoint", default=None,
                        help="Path to SAM checkpoint (.pth)")
    parser.add_argument("--sam-model-type", default="vit_b",
                        help="SAM model type (vit_b, vit_l, vit_h)")
    parser.add_argument("--owl-model", default="google/owlvit-base-patch32",
                        help="OWL-ViT model name from HuggingFace")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Logging interval (batches)")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    print("\n" + "="*70)
    print("OWL-SAM Adapter Training")
    print("="*70 + "\n")

    # Create config
    config = TrainingConfig(
        dataset_name="coco",
        dataset_split="train2017",
        num_images=args.num_images,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        mse_weight=args.mse_weight,
        cosine_weight=args.cosine_weight,
        device=args.device,
        fp16=args.fp16,
        checkpoint_dir=Path(args.checkpoint_dir),
        validation_size=args.validation_size,
        log_interval=args.log_interval,
    )

    print("Configuration:")
    print(f"  Dataset: COCO {config.dataset_split}")
    print(f"  Images: {config.num_images}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Device: {config.device}")
    print(f"  Loss: {config.mse_weight}*MSE + {config.cosine_weight}*Cosine")
    print()

    # Check COCO dataset
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"✗ Dataset directory not found: {dataset_dir}")
        print("\nTo download COCO:")
        print("  1. Go to https://cocodataset.org/")
        print("  2. Download train2017 and val2017")
        print("  3. Extract to data/coco/")
        print("\nOr use a small test dataset:")
        print("  mkdir -p data/coco/train2017")
        print("  mkdir -p data/coco/val2017")
        print("  # Copy some images there")
        sys.exit(1)

    # Create adapter
    print("Creating adapter...")
    adapter = OwlToSamAdapter()
    print(f"  Parameters: {adapter.get_parameter_count():,}")
    print(f"  Memory (FP32): {(adapter.get_parameter_count() * 4) / (1024**2):.2f}MB")
    print()

    # Create data loaders
    print("Creating data loaders...")
    train_loader = TrainingDataLoader.get_coco_loader(
        dataset_dir=args.dataset_dir,
        split="train2017",
        num_images=args.num_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=config.image_size,
    )

    val_loader = TrainingDataLoader.get_validation_loader(
        dataset_dir=args.dataset_dir,
        split="val2017",
        num_images=args.validation_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=config.image_size,
    )
    print()

    # Load encoders for Phase C
    owl_encoder, sam_encoder = None, None
    if args.phase_c:
        if not args.sam_checkpoint:
            print("✗ --sam-checkpoint is required for Phase C")
            sys.exit(1)
        from nanosam.encoders import OWLEncoder, SAMEncoder
        owl_encoder = OWLEncoder(model_name=args.owl_model).to(args.device)
        sam_encoder = SAMEncoder(checkpoint_path=args.sam_checkpoint, model_type=args.sam_model_type).to(args.device)
        print(f"  Phase C: real OWL + SAM encoders loaded\n")

    # Create trainer
    print("Creating trainer...")
    trainer = AdapterTrainer(
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        owl_encoder=owl_encoder,
        sam_encoder=sam_encoder,
    )
    print()

    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint: {args.load_checkpoint}")
        trainer.load_checkpoint(args.load_checkpoint)
        print()

    # Train
    print("Starting training...\n")
    history = trainer.train()

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
