#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark script for OWL to SAM adapter.

Tests latency, memory usage, and throughput on target hardware.
Run on Jetson Orin NX to get realistic performance metrics.
"""

import torch
import time
import psutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent))

from nanosam.adapter import OwlToSamAdapter


class AdapterBenchmark:
    """Comprehensive benchmark suite for adapter."""

    def __init__(self, device: str = "cuda", batch_sizes: List[int] = None):
        self.device = device
        self.batch_sizes = batch_sizes or [1, 2, 4, 8]
        self.adapter = OwlToSamAdapter().to(device)
        self.adapter.eval()
        self.results = {}

    def measure_latency(self, input_tensor: torch.Tensor, num_iterations: int = 100) -> Dict:
        """Measure inference latency."""
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.adapter(input_tensor)

        # Synchronize GPU
        if self.device == "cuda":
            torch.cuda.synchronize()

        # Measure
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.adapter(input_tensor)

        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        per_iter = (elapsed / num_iterations) * 1000  # ms

        return {
            "total_time_ms": elapsed * 1000,
            "per_iteration_ms": per_iter,
            "fps": 1000 / per_iter,
            "num_iterations": num_iterations,
        }

    def measure_memory(self, batch_size: int) -> Dict:
        """Measure peak memory usage."""
        torch.cuda.reset_peak_memory_stats()

        input_tensor = torch.randn(batch_size, 576, 768, device=self.device)

        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            output = self.adapter(input_tensor)

        if self.device == "cuda":
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        else:
            peak_memory = 0  # CPU memory tracking more complex

        input_size = input_tensor.numel() * 4 / (1024 ** 2)  # MB (FP32)
        output_size = output.numel() * 4 / (1024 ** 2)  # MB (FP32)

        return {
            "input_size_mb": input_size,
            "output_size_mb": output_size,
            "peak_memory_mb": peak_memory,
        }

    def benchmark_batch_sizes(self) -> Dict:
        """Benchmark different batch sizes."""
        results = {}

        for batch_size in self.batch_sizes:
            print(f"\nBenchmarking batch size {batch_size}...")
            input_tensor = torch.randn(batch_size, 576, 768, device=self.device)

            latency = self.measure_latency(input_tensor)
            memory = self.measure_memory(batch_size)

            results[f"batch_{batch_size}"] = {
                **latency,
                **memory,
                "throughput_samples_per_sec": latency["fps"] * batch_size,
            }

            print(f"  Latency: {latency['per_iteration_ms']:.2f}ms per batch")
            print(f"  FPS: {latency['fps']:.1f}")
            print(f"  Throughput: {results[f'batch_{batch_size}']['throughput_samples_per_sec']:.1f} samples/sec")
            print(f"  Peak Memory: {memory['peak_memory_mb']:.1f}MB")

        return results

    def benchmark_precision(self) -> Dict:
        """Benchmark different precision levels."""
        results = {}
        batch_size = 1
        input_tensor = torch.randn(batch_size, 576, 768, device=self.device)

        precisions = [
            ("FP32", torch.float32),
            ("FP16", torch.float16),
        ]

        for name, dtype in precisions:
            print(f"\nBenchmarking {name}...")
            adapter = self.adapter.to(dtype)
            input_tensor_typed = input_tensor.to(dtype)

            try:
                latency = self.measure_latency(input_tensor_typed, num_iterations=50)
                memory = self.measure_memory(batch_size)

                results[name] = {
                    **latency,
                    **memory,
                }
                print(f"  Latency: {latency['per_iteration_ms']:.2f}ms")
                print(f"  FPS: {latency['fps']:.1f}")
            except Exception as e:
                print(f"  Error: {e}")
                results[name] = {"error": str(e)}

        return results

    def analyze_computation(self) -> Dict:
        """Analyze computational complexity."""
        spatial_size = 64
        owl_dim = 768
        sam_dim = 256

        # Interpolation: approximate as 4 ops per element
        interp_ops = spatial_size * spatial_size * 4

        # MLP operations
        # Layer 1: 768 -> 512
        layer1_ops = spatial_size * spatial_size * owl_dim * 512
        # Layer 2: 512 -> 256
        layer2_ops = spatial_size * spatial_size * 512 * sam_dim

        total_flops = (interp_ops + layer1_ops + layer2_ops) / 1e9  # Giga FLOPs

        return {
            "interpolation_ops": int(interp_ops),
            "mlp_ops": int(layer1_ops + layer2_ops),
            "total_gflops": total_flops,
            "estimated_time_ms_at_40tflops": total_flops / 40,  # Orin NX ~40 TFLOPS
        }

    def get_system_info(self) -> Dict:
        """Get system information."""
        info = {
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "pytorch_version": torch.__version__,
        }

        if self.device == "cuda":
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = torch.cuda.get_device_capability(0)
            info["cuda_version"] = torch.version.cuda

        return info

    def run_full_benchmark(self) -> Dict:
        """Run all benchmarks."""
        print("\n" + "="*70)
        print("OWL to SAM Adapter Benchmark")
        print("="*70)

        all_results = {
            "system_info": self.get_system_info(),
            "adapter_config": {
                "parameters": self.adapter.get_parameter_count(),
                "parameter_memory_mb": (self.adapter.get_parameter_count() * 4) / (1024 ** 2),
            },
            "computation": self.analyze_computation(),
            "batch_size_benchmark": self.benchmark_batch_sizes(),
            "precision_benchmark": self.benchmark_precision(),
        }

        return all_results

    def save_results(self, results: Dict, output_path: str = None):
        """Save results to JSON file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"adapter_benchmark_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")

    def print_summary(self, results: Dict):
        """Print summary statistics."""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        # Model info
        print(f"\nModel Parameters: {results['adapter_config']['parameters']:,}")
        print(f"Parameter Memory (FP32): {results['adapter_config']['parameter_memory_mb']:.2f}MB")

        # Computational complexity
        comp = results['computation']
        print(f"\nComputational Complexity: {comp['total_gflops']:.2f} GFLOPs per frame")
        print(f"Estimated latency at 40 TFLOPS: {comp['estimated_time_ms_at_40tflops']:.2f}ms")

        # Batch size results
        print(f"\nLatency (Batch Size 1):")
        batch1_result = results['batch_size_benchmark'].get('batch_1', {})
        if batch1_result:
            print(f"  Per-frame latency: {batch1_result['per_iteration_ms']:.2f}ms")
            print(f"  Throughput: {batch1_result['throughput_samples_per_sec']:.1f} frames/sec")

        # Hardware utilization
        print(f"\nMemory Usage (Batch Size 1):")
        if batch1_result:
            print(f"  Input size: {batch1_result['input_size_mb']:.2f}MB")
            print(f"  Output size: {batch1_result['output_size_mb']:.2f}MB")
            print(f"  Peak activation: {batch1_result['peak_memory_mb']:.2f}MB")

        # Hospital robot context
        print(f"\nHospital Robot Context (Jetson Orin NX):")
        latency_ms = batch1_result.get('per_iteration_ms', 0)
        frames_per_sec = 1000 / latency_ms if latency_ms > 0 else 0
        print(f"  FPS achieved: {frames_per_sec:.1f}")
        print(f"  Requirement: 10 FPS")
        status = "✓ PASS" if frames_per_sec >= 10 else "✗ FAIL"
        print(f"  Status: {status}")

        print("\n" + "="*70)


def main():
    """Main benchmark entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark OWL to SAM adapter")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Batch sizes to benchmark")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    benchmark = AdapterBenchmark(device=args.device, batch_sizes=args.batch_sizes)
    results = benchmark.run_full_benchmark()
    benchmark.print_summary(results)
    benchmark.save_results(results, args.output)


if __name__ == "__main__":
    main()
