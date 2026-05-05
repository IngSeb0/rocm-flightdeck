"""
ROCm Readiness Scoring.

Rubric (max 100 points):
  Device abstraction         20
  Dependency compatibility   25
  Docker/runtime             20
  vLLM serving readiness     20
  Benchmark readiness        15

All scoring is deterministic — no random numbers, no LLM judgment.
"""

from __future__ import annotations

from typing import List

from core.models import Detection, ScoreBreakdown

# ---------------------------------------------------------------------------
# Category maximums
# ---------------------------------------------------------------------------
MAX_DEVICE = 20
MAX_DEPS = 25
MAX_DOCKER = 20
MAX_VLLM = 20
MAX_BENCH = 15

# ---------------------------------------------------------------------------
# Deduction table: detection_id -> (category, points, reason)
# ---------------------------------------------------------------------------
DEDUCTIONS = {
    "TORCH_CUDA":              ("device_abstraction",        10, "hardcoded torch.cuda reference"),
    "DOT_CUDA_CALL":           ("device_abstraction",         8, "hardcoded .cuda() call"),
    "TO_CUDA":                 ("device_abstraction",         4, ".to('cuda') / .to(\"cuda\") call"),
    "BITSANDBYTES":            ("dependency_compatibility",  10, "bitsandbytes (NVIDIA-only)"),
    "FLASH_ATTN":              ("dependency_compatibility",   7, "flash-attn (NVIDIA-optimized)"),
    "XFORMERS":                ("dependency_compatibility",   5, "xformers (NVIDIA-optimized)"),
    "TRITON":                  ("dependency_compatibility",   3, "triton (limited ROCm support)"),
    "NVIDIA_DOCKER":           ("docker_runtime",            12, "nvidia/cuda Docker base image"),
    "MISSING_DOCKERFILE_ROCM": ("docker_runtime",             8, "missing Dockerfile.rocm"),
    "MISSING_BENCHMARK_ROCM":  ("benchmark_readiness",       10, "missing benchmark_rocm.py"),
    "MISSING_README_AMD":      ("benchmark_readiness",        5, "missing README_AMD_MIGRATION.md"),
    "VLLM_MISSING":            ("vllm_serving",              15, "vLLM not detected"),
}


def compute_score(detections: List[Detection]) -> ScoreBreakdown:
    """
    Compute the ROCm Readiness Score from a list of detections.

    Returns a ScoreBreakdown with per-category scores, deductions list,
    and a plain-English explanation.
    """
    device = MAX_DEVICE
    deps = MAX_DEPS
    docker = MAX_DOCKER
    vllm = MAX_VLLM
    bench = MAX_BENCH

    deduction_messages: List[str] = []
    detected_ids = {d.id for d in detections}

    for det_id, (category, amount, reason) in DEDUCTIONS.items():
        if det_id not in detected_ids:
            continue
        deduction_messages.append(f"-{amount} {category}: {reason}")
        if category == "device_abstraction":
            device = max(0, device - amount)
        elif category == "dependency_compatibility":
            deps = max(0, deps - amount)
        elif category == "docker_runtime":
            docker = max(0, docker - amount)
        elif category == "vllm_serving":
            vllm = max(0, vllm - amount)
        elif category == "benchmark_readiness":
            bench = max(0, bench - amount)

    score = ScoreBreakdown(
        device_abstraction=device,
        dependency_compatibility=deps,
        docker_runtime=docker,
        vllm_serving=vllm,
        benchmark_readiness=bench,
        deductions=deduction_messages,
    )

    total = score.total_score
    if total >= 80:
        label = "Good — minor issues remain."
    elif total >= 60:
        label = "Fair — several blockers need attention before production migration."
    elif total >= 40:
        label = "Poor — significant NVIDIA/CUDA assumptions must be replaced."
    else:
        label = "Critical — repository is tightly coupled to NVIDIA/CUDA infrastructure."

    score.explanation = (
        f"Total ROCm Readiness Score: {total}/100. {label} "
        f"Category breakdown — "
        f"Device abstraction: {device}/{MAX_DEVICE}, "
        f"Dependency compatibility: {deps}/{MAX_DEPS}, "
        f"Docker/runtime: {docker}/{MAX_DOCKER}, "
        f"vLLM serving: {vllm}/{MAX_VLLM}, "
        f"Benchmark readiness: {bench}/{MAX_BENCH}."
    )

    return score
