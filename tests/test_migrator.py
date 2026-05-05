"""Tests for migrated repository generation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.migrator import migrate_repo
from core.patcher import generate_artifacts
from core.scanner import scan_repo
from core.detectors import run_all_detectors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAD_DEMO = os.path.join(ROOT, "demo_repos", "nvidia_locked_vllm_demo")


def test_migrator_creates_migrated_repo_and_improves_score(tmp_path):
    original_app = os.path.join(BAD_DEMO, "app.py")
    before_content = open(original_app, encoding="utf-8").read()
    artifacts = generate_artifacts(scan_repo(BAD_DEMO), str(tmp_path / "patches"))

    result = migrate_repo(BAD_DEMO, str(tmp_path / "migrated"), artifacts)

    assert os.path.isdir(result.migrated_repo_path)
    assert os.path.isfile(os.path.join(result.migrated_repo_path, "Dockerfile.rocm"))
    assert os.path.isfile(os.path.join(result.migrated_repo_path, "benchmark_rocm.py"))
    assert result.after_score.total_score > result.before_score.total_score
    assert result.after_score.total_score >= 85
    assert open(original_app, encoding="utf-8").read() == before_content


def test_migrated_python_uses_device_abstraction(tmp_path):
    artifacts = generate_artifacts(scan_repo(BAD_DEMO), str(tmp_path / "patches"))
    result = migrate_repo(BAD_DEMO, str(tmp_path / "migrated"), artifacts)
    migrated_app = open(os.path.join(result.migrated_repo_path, "app.py"), encoding="utf-8").read()

    assert ".cuda()" not in migrated_app
    assert '.to("cuda")' not in migrated_app
    assert ".to(device)" in migrated_app


def test_migrated_repo_is_rocm_first_and_legacy_safe(tmp_path):
    artifacts = generate_artifacts(scan_repo(BAD_DEMO), str(tmp_path / "patches"))
    result = migrate_repo(BAD_DEMO, str(tmp_path / "migrated"), artifacts)
    migrated = result.migrated_repo_path

    assert os.path.isfile(os.path.join(migrated, "Dockerfile"))
    assert os.path.isfile(os.path.join(migrated, "legacy", "Dockerfile.nvidia.legacy"))
    assert os.path.isfile(os.path.join(migrated, "legacy", "requirements.nvidia.legacy.txt"))
    assert os.path.isfile(os.path.join(migrated, "legacy", "benchmark.cuda.legacy.py"))

    dockerfile = open(os.path.join(migrated, "Dockerfile"), encoding="utf-8").read()
    requirements = open(os.path.join(migrated, "requirements.txt"), encoding="utf-8").read()
    assert "FROM rocm/pytorch:latest" in dockerfile
    assert "nvidia/cuda" not in dockerfile
    active_lines = [line.strip() for line in requirements.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert "bitsandbytes" not in active_lines
    assert "flash-attn" not in active_lines
    assert "xformers" not in active_lines
    assert "triton" not in active_lines

    ids = {d.id for d in run_all_detectors(scan_repo(migrated))}
    assert "NVIDIA_DOCKER" not in ids
    assert "BITSANDBYTES" not in ids
    assert "FLASH_ATTN" not in ids
    assert "XFORMERS" not in ids
    assert "TRITON" not in ids
