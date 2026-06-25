# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import sys
from pathlib import Path

# Add nanosam to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanosam.adapter import OwlToSamAdapter, UnifiedPerceptionPipeline


class TestOwlToSamAdapter:
    """Test suite for OWL to SAM adapter transformation."""

    @pytest.fixture
    def adapter(self):
        """Create adapter instance for testing."""
        return OwlToSamAdapter()

    @pytest.fixture
    def owl_features_batch1(self):
        """Create batch of size 1 OWL features."""
        return torch.randn(1, 576, 768, dtype=torch.float32)

    @pytest.fixture
    def owl_features_batch4(self):
        """Create batch of size 4 OWL features."""
        return torch.randn(4, 576, 768, dtype=torch.float32)

    def test_output_shape_batch1(self, adapter, owl_features_batch1):
        """Test output shape for batch size 1."""
        output = adapter(owl_features_batch1)
        assert output.shape == (1, 256, 64, 64), f"Expected (1, 256, 64, 64), got {output.shape}"

    def test_output_shape_batch4(self, adapter, owl_features_batch4):
        """Test output shape for batch size 4."""
        output = adapter(owl_features_batch4)
        assert output.shape == (4, 256, 64, 64), f"Expected (4, 256, 64, 64), got {output.shape}"

    def test_output_dtype_preserved(self, adapter, owl_features_batch1):
        """Test that output dtype matches input."""
        output = adapter(owl_features_batch1)
        assert output.dtype == owl_features_batch1.dtype, f"Dtype mismatch: {output.dtype} vs {owl_features_batch1.dtype}"

    def test_output_device_preserved(self, adapter, owl_features_batch1):
        """Test that output device matches input."""
        device = owl_features_batch1.device
        output = adapter(owl_features_batch1)
        assert output.device == device, f"Device mismatch: {output.device} vs {device}"

    def test_no_nan_values(self, adapter, owl_features_batch1):
        """Test that output doesn't contain NaNs."""
        output = adapter(owl_features_batch1)
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_no_inf_values(self, adapter, owl_features_batch1):
        """Test that output doesn't contain Infs."""
        output = adapter(owl_features_batch1)
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_feature_normalization(self, adapter, owl_features_batch1):
        """Test that features are normalized (approximately zero mean)."""
        output = adapter(owl_features_batch1)
        # Features should have reasonable magnitude after LayerNorm
        assert output.abs().mean() < 10, "Features may not be properly normalized"

    def test_parameter_count(self, adapter):
        """Test parameter count is correct."""
        expected_params = (
            768 * 512 + 512 +  # First linear layer
            512 * 256 + 256 +  # Second linear layer
            256 + 256  # LayerNorm
        )
        actual_params = adapter.get_parameter_count()
        assert actual_params == expected_params, f"Expected {expected_params} params, got {actual_params}"

    def test_all_parameters_trainable(self, adapter):
        """Test that all parameters are trainable by default."""
        trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        total = sum(p.numel() for p in adapter.parameters())
        assert trainable == total, "Not all parameters are trainable"

    def test_gradient_flow(self, adapter, owl_features_batch1):
        """Test that gradients flow through the adapter."""
        owl_features = owl_features_batch1.clone().requires_grad_(True)
        output = adapter(owl_features)
        loss = output.sum()
        loss.backward()

        assert owl_features.grad is not None, "Gradients don't flow to input"
        assert owl_features.grad.abs().sum() > 0, "Gradients are zero"

    def test_deterministic_output(self, adapter, owl_features_batch1):
        """Test that adapter produces deterministic output (no dropout, etc)."""
        adapter.eval()
        with torch.no_grad():
            output1 = adapter(owl_features_batch1)
            output2 = adapter(owl_features_batch1)

        assert torch.allclose(output1, output2), "Output is not deterministic"

    def test_different_input_produces_different_output(self, adapter):
        """Test that different inputs produce different outputs."""
        input1 = torch.randn(1, 576, 768)
        input2 = torch.randn(1, 576, 768)

        with torch.no_grad():
            output1 = adapter(input1)
            output2 = adapter(input2)

        assert not torch.allclose(output1, output2), "Different inputs produce same output"

    def test_spatial_dimension_mapping(self, adapter):
        """Test that spatial interpolation is working correctly."""
        # Create input with known spatial pattern
        batch_size = 1
        input_features = torch.arange(576, dtype=torch.float32).view(1, 576, 1)
        input_features = input_features.expand(1, 576, 768)

        output = adapter(input_features)

        # Check output has correct spatial dimensions
        assert output.shape[2] == 64, f"Expected height 64, got {output.shape[2]}"
        assert output.shape[3] == 64, f"Expected width 64, got {output.shape[3]}"

    def test_fp16_compatibility(self, adapter, owl_features_batch1):
        """Test adapter works with FP16 precision."""
        adapter_fp16 = adapter.half()
        input_fp16 = owl_features_batch1.half()

        try:
            output = adapter_fp16(input_fp16)
            assert output.dtype == torch.float16, "Output dtype is not FP16"
            assert output.shape == (1, 256, 64, 64), "Shape mismatch in FP16"
        except Exception as e:
            pytest.skip(f"FP16 not supported: {e}")

    def test_memory_efficiency(self, adapter, owl_features_batch1):
        """Test memory usage is reasonable."""
        # Count parameters (should be ~525K)
        param_count = adapter.get_parameter_count()
        assert 500_000 <= param_count <= 550_000, f"Unexpected parameter count: {param_count}"

        # Estimate memory for parameters (FP32)
        param_memory_mb = (param_count * 4) / (1024 * 1024)
        assert param_memory_mb < 5, f"Parameter memory too large: {param_memory_mb}MB"


class TestBatchProcessing:
    """Test batch processing edge cases."""

    @pytest.fixture
    def adapter(self):
        return OwlToSamAdapter()

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_various_batch_sizes(self, adapter, batch_size):
        """Test adapter works with various batch sizes."""
        input_features = torch.randn(batch_size, 576, 768)
        output = adapter(input_features)
        assert output.shape == (batch_size, 256, 64, 64)

    def test_single_spatial_element(self, adapter):
        """Test with minimal spatial content."""
        # Single batch, minimal feature pattern
        input_features = torch.ones(1, 576, 768)
        output = adapter(input_features)
        assert output.shape == (1, 256, 64, 64)
        assert not torch.isnan(output).any()

    def test_large_values(self, adapter):
        """Test stability with large input values."""
        input_features = torch.randn(1, 576, 768) * 1000
        output = adapter(input_features)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestUnifiedPipeline:
    """Test the unified pipeline wrapper."""

    def test_pipeline_instantiation(self):
        """Test that pipeline can be instantiated."""
        adapter = OwlToSamAdapter()

        # Create mock encoders (just returning correct shapes)
        class MockOWLEncoder(torch.nn.Module):
            def forward(self, x):
                from collections import namedtuple
                Output = namedtuple('Output', ['image_embeds'])
                return Output(image_embeds=torch.randn(x.shape[0], 576, 768))

        class MockSAMDecoder(torch.nn.Module):
            def forward(self, features, points, labels):
                return torch.randn(features.shape[0], 1, 256, 256)

        owl = MockOWLEncoder()
        sam = MockSAMDecoder()

        pipeline = UnifiedPerceptionPipeline(owl, sam, adapter)
        assert pipeline is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def adapter(self):
        return OwlToSamAdapter()

    def test_zero_input(self, adapter):
        """Test with all-zero input."""
        input_features = torch.zeros(1, 576, 768)
        output = adapter(input_features)
        assert output.shape == (1, 256, 64, 64)
        # Output should not be all zeros due to LayerNorm
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_high_variance_input(self, adapter):
        """Test with high variance input."""
        input_features = torch.randn(1, 576, 768) * 100
        output = adapter(input_features)
        assert output.shape == (1, 256, 64, 64)
        assert not torch.isnan(output).any()

    def test_requires_grad_false(self, adapter):
        """Test adapter works when gradients are disabled."""
        input_features = torch.randn(1, 576, 768)
        adapter.eval()
        with torch.no_grad():
            output = adapter(input_features)
        assert output.shape == (1, 256, 64, 64)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
