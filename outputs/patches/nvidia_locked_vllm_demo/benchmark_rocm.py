# =============================================================================
# AGENT-GENERATED FILE — ROCm FlightDeck starter template
# Review and adapt before using in production.
# Original files are NOT modified by this tool.
# =============================================================================
"""
benchmark_rocm.py — ROCm / MI300X benchmark methodology template.

IMPORTANT: This script does NOT report pre-filled numbers.
It provides the methodology and placeholder code that you run on real hardware.

Metrics to measure:
  - Latency          : milliseconds per output token (median / p95 / p99)
  - Throughput       : output tokens per second
  - VRAM usage       : peak HBM consumption in GB
  - Cold-start time  : seconds from process start to first token

Run on AMD Developer Cloud (MI300X) or your local ROCm GPU.
"""

import time
from typing import Optional

# ---------------------------------------------------------------------------
# NOTE: Install vLLM ROCm wheel before running this benchmark.
# ---------------------------------------------------------------------------
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def measure_vram_gb() -> Optional[float]:
    """Return current GPU VRAM usage in GB, or None if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
    except Exception:
        pass
    return None


def run_benchmark(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    prompts: list = None,
    max_tokens: int = 128,
    runs: int = 5,
) -> None:
    if prompts is None:
        prompts = [
            "Explain the attention mechanism in transformers.",
            "What is the difference between ROCm and CUDA?",
            "Describe the MI300X GPU architecture.",
        ]

    print("=" * 60)
    print("ROCm FlightDeck — Benchmark Methodology")
    print("=" * 60)
    print(f"Model       : {model_name}")
    print(f"Max tokens  : {max_tokens}")
    print(f"Runs        : {runs}")
    print(f"Prompts     : {len(prompts)}")
    print()

    if not VLLM_AVAILABLE:
        print("[SKIP] vLLM is not installed. Install the ROCm vLLM wheel and re-run.")
        print("  pip install vllm  # ROCm wheel")
        return

    sampling_params = SamplingParams(max_tokens=max_tokens)
    print("Loading model (cold start)...")
    cold_start = time.time()
    llm = LLM(model=model_name, device="cuda")
    cold_start_s = time.time() - cold_start
    print(f"Cold-start time: {cold_start_s:.2f}s")

    vram_after_load = measure_vram_gb()
    if vram_after_load:
        print(f"VRAM after load: {vram_after_load:.2f} GB")

    latencies = []
    for i in range(runs):
        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.time() - t0
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        latencies.append(elapsed)
        tps = total_tokens / elapsed
        print(f"Run {i+1}: {elapsed*1000:.1f} ms total | {tps:.1f} tokens/sec")

    latencies_ms = sorted(t * 1000 for t in latencies)
    print()
    print(f"Median latency : {latencies_ms[len(latencies_ms)//2]:.1f} ms")
    print(f"p95 latency    : {latencies_ms[int(len(latencies_ms)*0.95)]:.1f} ms")

    vram_peak = measure_vram_gb()
    if vram_peak:
        print(f"Peak VRAM      : {vram_peak:.2f} GB")

    print()
    print("Benchmark complete. Record results and compare against NVIDIA baseline.")


if __name__ == "__main__":
    run_benchmark()
