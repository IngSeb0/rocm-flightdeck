"""Tests for core/scoring.py."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detectors import run_all_detectors
from core.models import Detection
from core.scanner import scan_repo
from core.scoring import MAX_BENCH, MAX_DEPS, MAX_DEVICE, MAX_DOCKER, MAX_VLLM, compute_score

DEMO_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "demo_repos",
    "nvidia_locked_vllm_demo",
)


def test_score_below_70_for_demo_repo():
    """The intentionally broken demo repo must score below 70."""
    scan = scan_repo(DEMO_REPO)
    detections = run_all_detectors(scan)
    score = compute_score(detections)
    assert score.total_score < 70, (
        f"Expected score < 70 for the NVIDIA-locked demo repo, got {score.total_score}"
    )


def test_score_is_deterministic():
    """Scoring must return the same result on repeated calls."""
    scan = scan_repo(DEMO_REPO)
    detections = run_all_detectors(scan)
    score1 = compute_score(detections)
    score2 = compute_score(detections)
    assert score1.total_score == score2.total_score


def test_perfect_score_with_no_detections():
    """With no detections the score should be 100."""
    score = compute_score([])
    assert score.total_score == 100
    assert score.device_abstraction == MAX_DEVICE
    assert score.dependency_compatibility == MAX_DEPS
    assert score.docker_runtime == MAX_DOCKER
    assert score.vllm_serving == MAX_VLLM
    assert score.benchmark_readiness == MAX_BENCH


def test_score_never_goes_below_zero():
    """No category score should go below zero."""
    many = [
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
    score = compute_score(many)
    assert score.device_abstraction >= 0
    assert score.dependency_compatibility >= 0
    assert score.docker_runtime >= 0
    assert score.vllm_serving >= 0
    assert score.benchmark_readiness >= 0


def test_score_explanation_contains_total():
    """The explanation string must mention the total score."""
    scan = scan_repo(DEMO_REPO)
    detections = run_all_detectors(scan)
    score = compute_score(detections)
    assert str(score.total_score) in score.explanation


def test_deductions_list_populated():
    """Deductions list must be non-empty when there are detections."""
    detections = [
        Detection(id="NVIDIA_DOCKER", title="NVIDIA Docker", severity="critical"),
    ]
    score = compute_score(detections)
    assert len(score.deductions) > 0
