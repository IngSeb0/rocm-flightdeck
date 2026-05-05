"""
Patcher: generates reviewable ROCm migration artifacts.

All files are written to outputs/patches/<repo_name>/.
Original repository files are NEVER overwritten.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

from core.models import GeneratedArtifact, ScanResult

_AGENT_HEADER = (
    "# =============================================================================\n"
    "# AGENT-GENERATED FILE — ROCm FlightDeck starter template\n"
    "# Review and adapt before using in production.\n"
    "# Original files are NOT modified by this tool.\n"
    "# =============================================================================\n"
)


# ---------------------------------------------------------------------------
# Artifact content generators
# ---------------------------------------------------------------------------

def _dockerfile_rocm() -> str:
    return (
        _AGENT_HEADER
        + """
FROM rocm/pytorch:latest
# Alternatively: rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.1.0

LABEL maintainer="your-team"
LABEL description="ROCm-enabled vLLM inference service (MI300X)"

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \\
    python3-pip \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Python deps — ROCm-compatible
COPY requirements-rocm.txt ./
RUN pip install --no-cache-dir -r requirements-rocm.txt

COPY . .

# HIP device visibility (adjust for your hardware)
ENV HIP_VISIBLE_DEVICES=0
ENV ROCM_HOME=/opt/rocm

CMD ["python", "serve_vllm_rocm.py"]
"""
    )


def _requirements_rocm() -> str:
    return (
        _AGENT_HEADER
        + """# ROCm-compatible requirements
# Install the ROCm PyTorch wheel first:
#   pip install torch --index-url https://download.pytorch.org/whl/rocm6.1
#
# Then install vLLM ROCm build:
#   pip install vllm  # use the ROCm wheel from vLLM releases

torch  # install via ROCm wheel (see above)
vllm   # install via ROCm wheel

# Removed NVIDIA-only packages:
#   bitsandbytes  -> use AWQ/GPTQ or full-precision on MI300X
#   flash-attn    -> use PyTorch SDPA or flash-attention-rocm fork
#   xformers      -> not needed for vLLM ROCm
#   triton        -> use triton-rocm if needed, or composable_kernel

# Other deps
accelerate
transformers
"""
    )


def _serve_vllm_rocm() -> str:
    return (
        _AGENT_HEADER
        + '''"""
vLLM ROCm serving script — starter template for AMD MI300X.
Adapt model name, tensor-parallel degree, and quantization to your setup.
"""

import os
import subprocess
import sys


def main() -> None:
    model = os.environ.get("MODEL_NAME", "meta-llama/Llama-2-7b-hf")
    tp = os.environ.get("TENSOR_PARALLEL", "1")
    port = os.environ.get("PORT", "8000")

    # HIP device selection (set to match your MI300X slot)
    hip_devices = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    os.environ["HIP_VISIBLE_DEVICES"] = hip_devices

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--tensor-parallel-size", tp,
        "--port", port,
        "--device", "cuda",   # ROCm PyTorch exposes HIP GPUs as CUDA device
        "--dtype", "float16",  # or bfloat16 for MI300X
    ]

    print(f"Starting vLLM ROCm server: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
'''
    )


def _benchmark_rocm() -> str:
    return (
        _AGENT_HEADER
        + '''"""
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
'''
    )


def _readme_amd_migration(repo_name: str) -> str:
    return (
        _AGENT_HEADER
        + f"""# AMD ROCm Migration Guide — {repo_name}

> Auto-generated by ROCm FlightDeck. Review and expand before publishing.

## Overview
This document describes how to migrate `{repo_name}` from NVIDIA/CUDA to AMD ROCm,
targeting the MI300X GPU (192 GB HBM3).

## Prerequisites
- AMD GPU (MI300X recommended) with ROCm 6.x driver
- Docker with ROCm runtime (`--device /dev/kfd --device /dev/dri`)
- Python 3.10+

## ROCm Installation
```bash
# Ubuntu 22.04 — install ROCm 6.1
wget https://repo.radeon.com/amdgpu-install/6.1/ubuntu/jammy/amdgpu-install_6.1.60100-1_all.deb
sudo dpkg -i amdgpu-install_*.deb
sudo amdgpu-install --usecase=rocm
```

## Python Environment
```bash
# Install ROCm PyTorch wheel
pip install torch --index-url https://download.pytorch.org/whl/rocm6.1

# Install ROCm-compatible requirements
pip install -r requirements-rocm.txt
```

## Docker (ROCm)
```bash
docker build -f Dockerfile.rocm -t {repo_name}-rocm .
docker run --rm \\
  --device /dev/kfd \\
  --device /dev/dri \\
  --group-add video \\
  -p 8000:8000 \\
  {repo_name}-rocm
```

## Key Differences from NVIDIA Path
| Feature | NVIDIA | AMD ROCm |
|---------|--------|----------|
| Docker base | nvidia/cuda:12.x | rocm/pytorch:latest |
| Device string | `cuda` | `cuda` (via HIP backend) |
| Quantization | bitsandbytes | AWQ / GPTQ / FP8 |
| Flash attention | flash-attn | flash-attention-rocm / SDPA |
| Memory | up to 80 GB (H100) | 192 GB HBM3 (MI300X) |

## Known Limitations
- bitsandbytes quantization is not natively supported — see bitsandbytes-rocm fork
- Some triton kernels may require porting to HIP
- xformers is not required for vLLM on ROCm

## Running on AMD Developer Cloud
1. Provision a MI300X instance at cloud.amd.com
2. Clone this repository
3. Follow the Docker instructions above
4. Run `benchmark_rocm.py` to validate performance

## Benchmark Results
> Not yet measured. Run `benchmark_rocm.py` on MI300X and record results here.

| Metric | NVIDIA Baseline | AMD MI300X |
|--------|----------------|------------|
| Latency (p50) | TBD | TBD |
| Throughput (tok/s) | TBD | TBD |
| VRAM (GB) | TBD | TBD |
"""
    )


def _flightdeck_patch(scan: ScanResult) -> str:
    """Generate a unified-diff-style patch with suggested changes."""
    lines = [
        _AGENT_HEADER,
        "# flightdeck.patch — suggested changes for ROCm migration\n",
        "# Apply with: patch -p1 < flightdeck.patch\n",
        "# Review every hunk before applying.\n\n",
    ]

    # Find app.py or the first .py file to patch
    py_files = [p for p in scan.files_scanned if p.endswith(".py")]
    if py_files:
        target = py_files[0]
        lines.append(f"--- a/{target}\n")
        lines.append(f"+++ b/{target}\n")
        lines.append("@@ -1,5 +1,10 @@\n")
        lines.append("+import torch\n")
        lines.append("+\n")
        lines.append("+# ROCm FlightDeck: abstract device so code runs on both NVIDIA and AMD\n")
        lines.append("+device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n")
        lines.append("+\n")
        lines.append(" # ... existing code ...\n")
        lines.append("-# OLD: model.to('cuda')\n")
        lines.append("+# NEW: model.to(device)\n")
        lines.append("\n")

    # Dockerfile patch
    docker_files = [p for p in scan.files_scanned if "Dockerfile" in p and "rocm" not in p.lower()]
    if docker_files:
        target = docker_files[0]
        lines.append(f"--- a/{target}\n")
        lines.append(f"+++ b/{target}\n")
        lines.append("@@ -1,2 +1,3 @@\n")
        lines.append("-FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04\n")
        lines.append("+# Use Dockerfile.rocm for AMD GPU builds\n")
        lines.append("+FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04  # NVIDIA path\n")
        lines.append("+# FROM rocm/pytorch:latest                    # AMD ROCm path\n")

    return "".join(lines)


# ---------------------------------------------------------------------------
# Main patcher entry point
# ---------------------------------------------------------------------------

def generate_artifacts(scan: ScanResult, output_base: str) -> List[GeneratedArtifact]:
    """
    Generate all ROCm migration artifacts and write them to disk.

    Files are written to `output_base/<repo_name>/`.
    Returns a list of GeneratedArtifact objects.
    """
    repo_name = os.path.basename(os.path.abspath(scan.repo_path))
    # Sanitize to prevent path traversal in the output directory name
    repo_name = re.sub(r"[^\w\-.]", "_", repo_name) or "repo"
    # Use pathlib for safe path construction and bounds validation
    abs_output_base = Path(output_base).resolve()
    out_dir = abs_output_base / repo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_specs = [
        ("Dockerfile.rocm", _dockerfile_rocm()),
        ("requirements-rocm.txt", _requirements_rocm()),
        ("serve_vllm_rocm.py", _serve_vllm_rocm()),
        ("benchmark_rocm.py", _benchmark_rocm()),
        ("README_AMD_MIGRATION.md", _readme_amd_migration(repo_name)),
        ("flightdeck.patch", _flightdeck_patch(scan)),
    ]

    artifacts: List[GeneratedArtifact] = []
    for filename, content in artifact_specs:
        out_path = out_dir / filename
        # Confirm the target stays within our intended output directory
        try:
            out_path.relative_to(abs_output_base)
        except ValueError:
            continue  # skip any path that escapes the base (should not happen)
        out_path.write_text(content, encoding="utf-8")
        artifacts.append(GeneratedArtifact(
            filename=filename,
            content=content,
            output_path=str(out_path),
        ))

    return artifacts
