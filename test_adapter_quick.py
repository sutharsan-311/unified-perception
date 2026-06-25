#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quick validation script for adapter shape transformations.

Run this first to verify the adapter works with dummy tensors
before running full benchmarks on hardware.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nanosam.adapter import OwlToSamAdapter


def test_basic_transformation():
    """Test basic shape transformation."""
    print("\n" + "="*70)
    print("Testing OWL to SAM Adapter - Shape Transformation")
    print("="*70)

    adapter = OwlToSamAdapter()
    print(f"\n✓ Adapter created")
    print(f"  Parameters: {adapter.get_parameter_count():,}")
    print(f"  Memory (FP32): {(adapter.get_parameter_count() * 4) / (1024**2):.2f}MB")

    # Test batch size 1
    print(f"\nTest 1: Batch size = 1")
    owl_features = torch.randn(1, 576, 768)
    print(f"  Input shape:  {tuple(owl_features.shape)}")

    sam_features = adapter(owl_features)
    print(f"  Output shape: {tuple(sam_features.shape)}")

    expected_shape = (1, 256, 64, 64)
    assert sam_features.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}"
    print(f"  ✓ Shape correct!")

    # Test batch size 4
    print(f"\nTest 2: Batch size = 4")
    owl_features = torch.randn(4, 576, 768)
    print(f"  Input shape:  {tuple(owl_features.shape)}")

    sam_features = adapter(owl_features)
    print(f"  Output shape: {tuple(sam_features.shape)}")

    expected_shape = (4, 256, 64, 64)
    assert sam_features.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}"
    print(f"  ✓ Shape correct!")

    # Test feature validity
    print(f"\nTest 3: Feature Quality Checks")
    owl_features = torch.randn(1, 576, 768)
    sam_features = adapter(owl_features)

    has_nan = torch.isnan(sam_features).any()
    has_inf = torch.isinf(sam_features).any()
    mean_val = sam_features.mean().item()
    std_val = sam_features.std().item()

    print(f"  NaN check: {'✗ FAIL' if has_nan else '✓ PASS'}")
    print(f"  Inf check: {'✗ FAIL' if has_inf else '✓ PASS'}")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Std:  {std_val:.4f}")

    assert not has_nan, "Output contains NaN!"
    assert not has_inf, "Output contains Inf!"
    print(f"  ✓ All quality checks passed!")

    # Test gradient flow
    print(f"\nTest 4: Gradient Flow")
    owl_features = torch.randn(1, 576, 768, requires_grad=True)
    sam_features = adapter(owl_features)
    loss = sam_features.sum()
    loss.backward()

    has_grad = owl_features.grad is not None
    grad_nonzero = owl_features.grad.abs().sum().item() > 0

    print(f"  Gradient received: {'✓ PASS' if has_grad else '✗ FAIL'}")
    print(f"  Gradient non-zero: {'✓ PASS' if grad_nonzero else '✗ FAIL'}")

    assert has_grad, "No gradients received!"
    assert grad_nonzero, "Gradients are zero!"
    print(f"  ✓ Gradient flow verified!")

    # Test different inputs
    print(f"\nTest 5: Different Inputs → Different Outputs")
    input1 = torch.randn(1, 576, 768)
    input2 = torch.randn(1, 576, 768)

    with torch.no_grad():
        output1 = adapter(input1)
        output2 = adapter(input2)

    outputs_differ = not torch.allclose(output1, output2, atol=1e-6)
    print(f"  Outputs differ: {'✓ PASS' if outputs_differ else '✗ FAIL'}")
    assert outputs_differ, "Different inputs produce same output!"

    print(f"\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)

    return True


def test_gpu_compatibility():
    """Test GPU compatibility if CUDA available."""
    if not torch.cuda.is_available():
        print("\nℹ CUDA not available, skipping GPU tests")
        return True

    print("\n" + "="*70)
    print("Testing GPU Compatibility")
    print("="*70)

    adapter = OwlToSamAdapter()
    device = "cuda"

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")

    # Move to GPU
    adapter = adapter.to(device)
    owl_features = torch.randn(1, 576, 768, device=device)

    print(f"  Adapter moved to GPU: ✓")
    print(f"  Input moved to GPU: ✓")

    # Test inference
    sam_features = adapter(owl_features)
    print(f"  Inference completed: ✓")

    # Check output is on GPU
    on_gpu = sam_features.device.type == "cuda"
    print(f"  Output on GPU: {'✓' if on_gpu else '✗'}")

    assert on_gpu, "Output not on GPU!"
    assert sam_features.shape == (1, 256, 64, 64), "Shape mismatch!"

    print(f"\n✓ GPU compatibility verified!")

    return True


def test_throughput_estimate():
    """Estimate throughput on current hardware."""
    print("\n" + "="*70)
    print("Estimating Throughput")
    print("="*70)

    adapter = OwlToSamAdapter()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = adapter.to(device)

    # Quick latency measurement
    import time

    owl_features = torch.randn(1, 576, 768, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = adapter(owl_features)

    if device == "cuda":
        torch.cuda.synchronize()

    # Measure
    iterations = 100
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = adapter(owl_features)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    per_frame_ms = (elapsed / iterations) * 1000
    fps = 1000 / per_frame_ms

    print(f"\nMeasurement (100 iterations):")
    print(f"  Device: {device}")
    print(f"  Per-frame latency: {per_frame_ms:.2f}ms")
    print(f"  Throughput: {fps:.1f} FPS")

    print(f"\nHospital Robot Context:")
    print(f"  Requirement: ≥10 FPS")
    print(f"  Current: {fps:.1f} FPS")
    status = "✓ PASS" if fps >= 10 else "⚠ WARN"
    print(f"  Status: {status}")

    print(f"\n" + "="*70)

    return True


def main():
    """Run all quick tests."""
    try:
        print("\n🧪 Running quick adapter validation tests...\n")

        test_basic_transformation()
        test_gpu_compatibility()
        test_throughput_estimate()

        print("\n" + "="*70)
        print("✓ ALL VALIDATION TESTS PASSED")
        print("="*70)
        print("\nAdapter is ready for Phase B training!")
        print("Next steps:")
        print("  1. Run full benchmark: python benchmark_adapter.py")
        print("  2. Prepare training data")
        print("  3. Implement training loop")
        print("="*70 + "\n")

        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
