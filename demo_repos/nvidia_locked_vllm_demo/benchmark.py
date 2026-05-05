"""
Benchmark script — NVIDIA/CUDA version (demo repo).
This intentionally uses CUDA-specific assumptions.
"""

import time
import torch


def benchmark_inference():
    """Run a naive benchmark on CUDA."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    device = torch.device("cuda")
    print(f"Running benchmark on {torch.cuda.get_device_name(0)}")

    # Simulate some tensor operations
    x = torch.randn(1024, 1024).cuda()
    t0 = time.time()
    for _ in range(100):
        x = x @ x
    elapsed = time.time() - t0
    print(f"Matrix multiply (1024x1024 x100): {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    benchmark_inference()
