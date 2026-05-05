"""Migration planner: generates ordered migration steps from detections."""

from __future__ import annotations

from typing import List

from core.models import Detection, MigrationPlan


# Map detection IDs to migration step text
_STEP_MAP = {
    "TORCH_CUDA": (
        "1. Replace hardcoded torch.cuda references with device abstraction. "
        "Use `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` "
        "and pass `device` throughout the model loading and inference code."
    ),
    "DOT_CUDA_CALL": (
        "2. Replace .cuda() calls with .to(device). "
        "Note: ROCm PyTorch exposes AMD GPUs through the CUDA-compatible API, so "
        ".to('cuda') works on MI300X when using the ROCm PyTorch wheel — "
        "but abstracting the device string improves portability."
    ),
    "TO_CUDA": (
        "3. Keep .to('cuda') calls where ROCm PyTorch is used (they work on MI300X). "
        "For maximum portability, wrap in a device variable: "
        "`device = 'cuda'` and use `.to(device)`."
    ),
    "NVIDIA_DOCKER": (
        "4. Replace the nvidia/cuda Docker base image with an AMD ROCm image. "
        "Use `rocm/pytorch:latest` or `rocm/rocm-terminal` as the base. "
        "See the generated Dockerfile.rocm in outputs/patches/<repo>/."
    ),
    "MISSING_DOCKERFILE_ROCM": (
        "5. Create Dockerfile.rocm alongside the existing Dockerfile. "
        "A starter template has been generated at outputs/patches/<repo>/Dockerfile.rocm."
    ),
    "CUDA_HOME": (
        "6. Replace CUDA_HOME environment variable references with ROCM_HOME or ROCM_PATH "
        "in build scripts, Dockerfiles, and shell scripts."
    ),
    "NVCC": (
        "7. Replace nvcc compiler invocations with hipcc for AMD GPU kernels. "
        "For PyTorch C++ extensions, use the HIP-compatible torch.utils.cpp_extension build path."
    ),
    "BITSANDBYTES": (
        "8. Remove or replace bitsandbytes. "
        "Options: (a) bitsandbytes-rocm community fork, "
        "(b) switch to AWQ or GPTQ quantization (ROCm-compatible), "
        "(c) for MI300X with 192 GB HBM consider full-precision or FP8 without quantization."
    ),
    "FLASH_ATTN": (
        "9. Replace flash-attn with a ROCm-compatible attention implementation. "
        "Options: (a) flash-attention-rocm fork, "
        "(b) torch.nn.functional.scaled_dot_product_attention (SDPA) — has ROCm backend."
    ),
    "XFORMERS": (
        "10. Remove xformers or replace with ROCm-native attention. "
        "vLLM on ROCm uses built-in attention backends; xformers is not required. "
        "Use PyTorch SDPA or composable_kernel-based kernels for custom attention."
    ),
    "TRITON": (
        "11. Audit triton usage. OpenAI Triton has experimental ROCm/HIP support (triton-rocm). "
        "Test each Triton kernel individually. AMD composable_kernel is the production alternative."
    ),
    "VLLM_PRESENT": (
        "12. Build or install the ROCm-enabled vLLM wheel. "
        "Use the generated serve_vllm_rocm.py script as a starting point. "
        "Pass `--device rocm` or set HIP_VISIBLE_DEVICES as appropriate."
    ),
    "VLLM_MISSING": (
        "12. If vLLM serving is intended, add vllm to requirements-rocm.txt and "
        "install the ROCm wheel. See outputs/patches/<repo>/serve_vllm_rocm.py."
    ),
    "MISSING_BENCHMARK_ROCM": (
        "13. Add benchmark_rocm.py to measure performance on AMD MI300X. "
        "Measure: latency (ms/token), throughput (tokens/sec), VRAM usage (GB), cold-start time. "
        "A methodology template is in outputs/patches/<repo>/benchmark_rocm.py."
    ),
    "MISSING_README_AMD": (
        "14. Add README_AMD_MIGRATION.md documenting: "
        "ROCm version requirements, driver setup, differences from NVIDIA path, "
        "known limitations, and how to run on AMD Developer Cloud / MI300X. "
        "A starter is in outputs/patches/<repo>/README_AMD_MIGRATION.md."
    ),
}

_ALWAYS_FIRST = (
    "0. Pin ROCm version: decide on ROCm 6.x for MI300X compatibility. "
    "Verify all dependencies against the ROCm compatibility matrix."
)

_ALWAYS_LAST = (
    "15. Test the migrated stack on AMD Developer Cloud (MI300X) or a local ROCm GPU. "
    "Run benchmark_rocm.py and compare against the NVIDIA baseline."
)


def generate_plan(detections: List[Detection]) -> MigrationPlan:
    """Generate an ordered migration plan from a list of detections."""
    detected_ids = {d.id for d in detections}
    steps: List[str] = [_ALWAYS_FIRST]

    for det_id, step_text in _STEP_MAP.items():
        if det_id in detected_ids:
            steps.append(step_text)

    steps.append(_ALWAYS_LAST)
    return MigrationPlan(steps=steps)
