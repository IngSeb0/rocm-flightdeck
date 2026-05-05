"""Tests for deterministic ROCm Readiness Scoring."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detectors import run_all_detectors
from core.models import Detection
from core.scanner import scan_repo
from core.scoring import MAX_BENCH, MAX_DEPS, MAX_DEVICE, MAX_DOCKER, MAX_VLLM, assessment_label, compute_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _score_demo(name: str):
    scan = scan_repo(os.path.join(ROOT, "demo_repos", name))
    return compute_score(run_all_detectors(scan))


def test_demo_score_ordering():
    bad = _score_demo("nvidia_locked_vllm_demo")
    medium = _score_demo("partially_portable_pytorch_demo")
    ready = _score_demo("rocm_ready_vllm_demo")

    assert bad.total_score < medium.total_score < ready.total_score
    assert bad.total_score <= 30
    assert 55 <= medium.total_score <= 75
    assert ready.total_score >= 85


def test_score_is_deterministic():
    first = _score_demo("nvidia_locked_vllm_demo")
    second = _score_demo("nvidia_locked_vllm_demo")
    assert first.total_score == second.total_score
    assert first.deductions == second.deductions


def test_perfect_score_with_no_detections():
    score = compute_score([])
    assert score.total_score == 100
    assert score.device_abstraction == MAX_DEVICE
    assert score.dependency_compatibility == MAX_DEPS
    assert score.docker_runtime == MAX_DOCKER
    assert score.vllm_serving == MAX_VLLM
    assert score.benchmark_readiness == MAX_BENCH


def test_score_never_goes_below_zero():
    detections = [
        Detection(id="TORCH_CUDA", title="t", severity="critical"),
        Detection(id="DOT_CUDA_CALL", title="t", severity="critical"),
        Detection(id="TO_CUDA", title="t", severity="warning"),
        Detection(id="BITSANDBYTES", title="t", severity="critical"),
        Detection(id="FLASH_ATTN", title="t", severity="critical"),
        Detection(id="XFORMERS", title="t", severity="critical"),
        Detection(id="TRITON", title="t", severity="warning"),
        Detection(id="NVIDIA_DOCKER", title="t", severity="critical"),
        Detection(id="MISSING_DOCKERFILE_ROCM", title="t", severity="warning"),
        Detection(id="MISSING_BENCHMARK_ROCM", title="t", severity="info"),
        Detection(id="MISSING_README_AMD", title="t", severity="info"),
        Detection(id="VLLM_MISSING", title="t", severity="info"),
    ]
    score = compute_score(detections)
    assert score.device_abstraction >= 0
    assert score.dependency_compatibility >= 0
    assert score.docker_runtime >= 0
    assert score.vllm_serving >= 0
    assert score.benchmark_readiness >= 0


def test_deductions_are_explainable():
    score = compute_score([
        Detection(
            id="NVIDIA_DOCKER",
            title="NVIDIA Docker",
            severity="critical",
            file_path="Dockerfile",
            recommendation="Use Dockerfile.rocm.",
        )
    ])
    assert score.deductions
    deduction = score.deductions[0]
    assert deduction["reason"]
    assert deduction["severity"] == "critical"
    assert deduction["file"] == "Dockerfile"
    assert deduction["recommendation"]


def test_assessment_labels():
    assert assessment_label(20) == "Critical - NVIDIA/CUDA locked"
    assert assessment_label(45) == "Partial - significant migration required"
    assert assessment_label(70) == "Reviewable - migration artifacts generated, validation required"
    assert assessment_label(90) == "ROCm Candidate - ready for MI300X validation"
    assert assessment_label(100) == "ROCm Ready - no major static blockers detected"
