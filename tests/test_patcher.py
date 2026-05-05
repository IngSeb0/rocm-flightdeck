"""Tests for core/patcher.py."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.patcher import generate_artifacts
from core.scanner import scan_repo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAD_DEMO = os.path.join(ROOT, "demo_repos", "nvidia_locked_vllm_demo")


def test_patcher_generates_all_expected_artifacts(tmp_path):
    expected = {
        "Dockerfile.rocm",
        "requirements-rocm.txt",
        "serve_vllm_rocm.py",
        "benchmark_rocm.py",
        "README_AMD_MIGRATION.md",
        "flightdeck.patch",
    }
    artifacts = generate_artifacts(scan_repo(BAD_DEMO), str(tmp_path))

    assert {artifact.filename for artifact in artifacts} == expected
    for artifact in artifacts:
        assert os.path.isfile(artifact.output_path)


def test_patcher_benchmark_contains_methodology_not_fake_numbers(tmp_path):
    artifacts = generate_artifacts(scan_repo(BAD_DEMO), str(tmp_path))
    benchmark = next(a for a in artifacts if a.filename == "benchmark_rocm.py")

    assert "tokens_per_second = " not in benchmark.content
    assert "latency_ms = " not in benchmark.content
    assert "methodology" in benchmark.content.lower()


def test_flightdeck_patch_contains_real_hunks(tmp_path):
    artifacts = generate_artifacts(scan_repo(BAD_DEMO), str(tmp_path))
    patch = next(a for a in artifacts if a.filename == "flightdeck.patch").content

    assert "--- a/app.py" in patch
    assert '+model.to(device)' in patch
    assert '+inputs["input_ids"].to(device)' in patch
    assert "--- a/benchmark.py" in patch
    assert "+x = torch.randn(1024, 1024).to(device)" in patch
    assert "--- a/Dockerfile" in patch
    assert "Use Dockerfile.rocm for AMD ROCm builds" in patch


def test_patcher_does_not_overwrite_originals(tmp_path):
    artifacts = generate_artifacts(scan_repo(BAD_DEMO), str(tmp_path))
    for artifact in artifacts:
        assert artifact.output_path.startswith(str(tmp_path))
        assert not artifact.output_path.startswith(BAD_DEMO)
