# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class OwlToSamAdapter(nn.Module):
    """
    Adapter layer that transforms OWL-ViT-B encoder output to SAM-compatible features.

    Converts:
        OWL output:  (B, 576, 768)   [576 tokens, 768-dim features]
        SAM input:   (B, 256, 64, 64) [64x64 spatial, 256-dim features]

    Architecture:
        1. Reshape to 2D spatial grid (24x24)
        2. Bilinear interpolate to target spatial size (64x64)
        3. Reduce features via 2-layer MLP (768 -> 512 -> 256)
        4. LayerNorm for stability
        5. Permute to channel-first format for SAM
    """

    def __init__(
        self,
        owl_hidden_dim: int = 768,
        sam_hidden_dim: int = 256,
        mlp_hidden_dim: int = 512,
        target_spatial_size: int = 64,
        owl_patch_size: int = 32,
        sam_patch_size: int = 16,
        owl_image_size: int = 768,
        sam_image_size: int = 1024,
    ):
        """
        Args:
            owl_hidden_dim: OWL-ViT output feature dimension (768)
            sam_hidden_dim: SAM expected feature dimension (256)
            mlp_hidden_dim: Hidden dimension in MLP (512)
            target_spatial_size: Target spatial resolution (64)
            owl_patch_size: OWL patch size in pixels (32)
            sam_patch_size: SAM patch size in pixels (16)
            owl_image_size: OWL input image size (768)
            sam_image_size: SAM input image size (1024)
        """
        super().__init__()

        self.owl_hidden_dim = owl_hidden_dim
        self.sam_hidden_dim = sam_hidden_dim
        self.target_spatial_size = target_spatial_size

        # Calculate spatial dimensions
        self.owl_spatial_size = owl_image_size // owl_patch_size  # 768/32 = 24
        self.sam_spatial_size = sam_image_size // sam_patch_size  # 1024/16 = 64

        # Feature transformation: 768 -> 512 -> 256
        self.mlp = nn.Sequential(
            nn.Linear(owl_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, sam_hidden_dim)
        )

        # Stability layer
        self.norm = nn.LayerNorm(sam_hidden_dim)

    def forward(self, owl_features: torch.Tensor) -> torch.Tensor:
        """
        Transform OWL features to SAM-compatible format.

        Args:
            owl_features: (batch_size, 576, 768)
                - 576 = num_patches (24 x 24)
                - 768 = hidden dimension

        Returns:
            sam_features: (batch_size, 256, 64, 64)
                - 256 = channel dimension
                - 64 x 64 = spatial grid
        """
        batch_size = owl_features.shape[0]

        # Step 1: Reshape from sequence to 2D spatial grid
        # (B, 576, 768) -> (B, 24, 24, 768)
        owl_grid = owl_features.view(
            batch_size,
            self.owl_spatial_size,
            self.owl_spatial_size,
            self.owl_hidden_dim
        )

        # Step 2: Spatial interpolation (bilinear)
        # (B, 24, 24, 768) -> (B, 768, 24, 24) for interpolation
        owl_grid = owl_grid.permute(0, 3, 1, 2)

        upsampled = F.interpolate(
            owl_grid,
            size=(self.target_spatial_size, self.target_spatial_size),
            mode='bilinear',
            align_corners=True
        )
        # (B, 768, 64, 64)

        # Step 3: Feature dimension reduction via MLP
        # Reshape for batch processing: (B, 768, 64, 64) -> (B*64*64, 768)
        B, C, H, W = upsampled.shape
        upsampled_flat = upsampled.permute(0, 2, 3, 1).contiguous()
        upsampled_flat = upsampled_flat.view(B * H * W, C)

        # Apply MLP
        reduced = self.mlp(upsampled_flat)  # (B*64*64, 256)

        # Step 4: Normalize features
        reduced = self.norm(reduced)

        # Step 5: Reshape back to spatial format
        # (B*64*64, 256) -> (B, 64, 64, 256) -> (B, 256, 64, 64)
        sam_features = reduced.view(B, H, W, self.sam_hidden_dim)
        sam_features = sam_features.permute(0, 3, 1, 2).contiguous()

        return sam_features

    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_flops_per_image(self, batch_size: int = 1) -> int:
        """
        Estimate FLOPs per image (rough approximation).

        Returns:
            Estimated FLOPs for processing one image
        """
        spatial_elements = self.target_spatial_size * self.target_spatial_size

        # Interpolation (approximate as 4 memory ops per element)
        interp_flops = spatial_elements * 4

        # MLP forward passes
        mlp_flops = spatial_elements * (
            self.owl_hidden_dim * 512 * 2 +  # two linear layers, approximate multiplications
            512 + 256  # bias additions
        )

        return int(interp_flops + mlp_flops)


class UnifiedPerceptionPipeline(nn.Module):
    """
    Unified pipeline combining OWL detection + SAM segmentation.

    This is a simple wrapper showing how the adapter fits into the full system.
    Actual integration will be in segment_from_owl.py example.
    """

    def __init__(
        self,
        owl_encoder,
        sam_mask_decoder,
        adapter: OwlToSamAdapter,
        device: str = "cuda"
    ):
        super().__init__()
        self.owl_encoder = owl_encoder
        self.sam_decoder = sam_mask_decoder
        self.adapter = adapter
        self.device = device

    def forward(
        self,
        image: torch.Tensor,
        points: torch.Tensor = None,
        point_labels: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single forward pass for detection + segmentation.

        Args:
            image: (B, 3, 768, 768) for OWL input
            points: Detection points for SAM
            point_labels: Point labels for SAM

        Returns:
            masks, scores
        """
        # OWL detection
        owl_output = self.owl_encoder(image)
        owl_features = owl_output.image_embeds  # (B, 576, 768)

        # Transform features
        sam_features = self.adapter(owl_features)  # (B, 256, 64, 64)

        # SAM segmentation (simplified - actual use includes prompt encoder)
        if points is not None:
            masks = self.sam_decoder(
                sam_features,
                points,
                point_labels
            )
            return masks

        return sam_features


if __name__ == "__main__":
    # Quick sanity check
    print("Testing OwlToSamAdapter...")

    adapter = OwlToSamAdapter()
    print(f"  Parameters: {adapter.get_parameter_count():,}")
    print(f"  FLOPs per image: {adapter.get_flops_per_image():,}")

    # Test with dummy tensors
    batch_size = 1
    owl_features = torch.randn(batch_size, 576, 768).cuda()

    sam_features = adapter(owl_features)
    print(f"\n  OWL input shape:  {owl_features.shape}")
    print(f"  SAM output shape: {sam_features.shape}")
    print(f"  Expected shape:   torch.Size([{batch_size}, 256, 64, 64])")

    assert sam_features.shape == (batch_size, 256, 64, 64), "Shape mismatch!"
    print("\n✓ Shape transformation verified!")
