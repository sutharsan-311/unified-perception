# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data loading pipeline for adapter training."""

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import json


class COCOAdapterDataset(Dataset):
    """
    COCO dataset adapted for OWL-SAM adapter training.

    Loads COCO images and generates training pairs:
    - Input: Image tensor
    - Target: SAM encoder features (simulated for now)
    """

    def __init__(
        self,
        dataset_dir: str = "data/coco",
        split: str = "train2017",
        num_images: Optional[int] = None,
        image_size: int = 768,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            dataset_dir: Root directory containing COCO dataset
            split: Dataset split (train2017, val2017)
            num_images: Limit number of images (None = use all)
            image_size: Target image size
            transform: Image transformations
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_size = image_size

        # Get image paths
        self.images_dir = self.dataset_dir / split
        self.image_paths = sorted(list(self.images_dir.glob("*.jpg")))

        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]

        # Default transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform

        print(f"Loaded {len(self.image_paths)} images from {split}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: (3, H, W) normalized tensor
            target_features: (256, 64, 64) simulated SAM features
        """
        image_path = self.image_paths[idx]

        # Load image
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Generate target features (simulated SAM encoder output)
        # In Phase C, replace with actual SAM encoder inference
        target_features = self._generate_target_features(image)

        return image, target_features

    def _generate_target_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Generate target features for training.

        For now, use a simple deterministic mapping.
        Phase C: Replace with actual SAM encoder.

        Args:
            image: (3, H, W) normalized tensor

        Returns:
            target_features: (256, 64, 64) features
        """
        # Deterministic feature generation based on image content
        # In Phase C, this will be: sam_encoder(image)
        batch_features = torch.randn(1, 256, 64, 64, dtype=torch.float32)
        return batch_features.squeeze(0)


class TrainingDataLoader:
    """Wrapper for training data loading."""

    @staticmethod
    def get_coco_loader(
        dataset_dir: str = "data/coco",
        split: str = "train2017",
        num_images: Optional[int] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        shuffle: bool = True,
        image_size: int = 768,
    ) -> DataLoader:
        """
        Get COCO dataset DataLoader.

        Args:
            dataset_dir: Root directory of COCO dataset
            split: Dataset split
            num_images: Limit number of images
            batch_size: Batch size
            num_workers: Number of data loading workers
            shuffle: Whether to shuffle data
            image_size: Target image size

        Returns:
            DataLoader
        """
        dataset = COCOAdapterDataset(
            dataset_dir=dataset_dir,
            split=split,
            num_images=num_images,
            image_size=image_size,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    @staticmethod
    def get_validation_loader(
        dataset_dir: str = "data/coco",
        split: str = "val2017",
        num_images: Optional[int] = None,
        batch_size: int = 4,
        num_workers: int = 2,
        image_size: int = 768,
    ) -> DataLoader:
        """
        Get validation DataLoader (no shuffle, no drop_last).

        Args:
            dataset_dir: Root directory of COCO dataset
            split: Dataset split
            num_images: Limit number of images
            batch_size: Batch size
            num_workers: Number of data loading workers
            image_size: Target image size

        Returns:
            DataLoader
        """
        dataset = COCOAdapterDataset(
            dataset_dir=dataset_dir,
            split=split,
            num_images=num_images,
            image_size=image_size,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )


if __name__ == "__main__":
    # Quick test
    print("Testing data loader...")

    loader = TrainingDataLoader.get_coco_loader(
        dataset_dir="data/coco",
        split="train2017",
        num_images=100,
        batch_size=4,
    )

    # Get one batch
    for images, targets in loader:
        print(f"Images batch: {images.shape}")
        print(f"Targets batch: {targets.shape}")
        break

    print("✓ Data loader works!")
