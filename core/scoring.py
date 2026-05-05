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


def assessment_label(total: int) -> str:
    """Return deterministic readiness label for a score total."""
    if total <= 30:
        return "Critical - NVIDIA/CUDA locked"
    if total <= 60:
        return "Partial - significant migration required"
    if total <= 80:
        return "Reviewable - migration artifacts generated, validation required"
    if total <= 94:
        return "ROCm Candidate - ready for MI300X validation"
    return "ROCm Ready - no major static blockers detected"


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

    deduction_messages: List[dict[str, str]] = []
    detected_ids = {d.id for d in detections}

    for det_id, (category, amount, reason) in DEDUCTIONS.items():
        if det_id not in detected_ids:
            continue
        related = next(d for d in detections if d.id == det_id)
        deduction_messages.append({
            "points": str(amount),
            "category": category,
            "reason": reason,
            "severity": related.severity,
            "file": related.file_path or "",
            "recommendation": related.recommendation,
        })
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
    label = assessment_label(total)

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
