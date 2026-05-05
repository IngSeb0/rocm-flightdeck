"""Tests for core/patcher.py and core/detectors.py."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detectors import (
    detect_bitsandbytes,
    detect_cuda_call,
    detect_flash_attn,
    detect_nvidia_docker,
    detect_to_cuda,
    detect_torch_cuda,
    detect_triton,
    detect_xformers,
    run_all_detectors,
)
from core.models import ScanResult
from core.patcher import generate_artifacts
from core.scanner import scan_repo

DEMO_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "demo_repos",
    "nvidia_locked_vllm_demo",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scan(**files) -> ScanResult:
    """Build a minimal ScanResult from keyword-arg file contents."""
    scan = ScanResult(repo_path="/fake")
    for path, content in files.items():
        scan.files_scanned.append(path)
        scan.file_contents[path] = content
    return scan


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------

def test_detect_torch_cuda():
    scan = _make_scan(**{"app.py": "import torch\ntorch.cuda.is_available()\n"})
    hits = detect_torch_cuda(scan)
    assert len(hits) == 1
    assert hits[0].id == "TORCH_CUDA"
    assert hits[0].severity == "critical"


def test_detect_cuda_call():
    scan = _make_scan(**{"model.py": "model.cuda()\n"})
    hits = detect_cuda_call(scan)
    assert len(hits) == 1
    assert hits[0].id == "DOT_CUDA_CALL"


def test_detect_to_cuda_double_quote():
    scan = _make_scan(**{"train.py": 'model.to("cuda")\n'})
    hits = detect_to_cuda(scan)
    assert len(hits) == 1
    assert hits[0].id == "TO_CUDA"


def test_detect_to_cuda_single_quote():
    scan = _make_scan(**{"train.py": "model.to('cuda')\n"})
    hits = detect_to_cuda(scan)
    assert len(hits) == 1
    assert hits[0].id == "TO_CUDA"


def test_detect_nvidia_docker():
    scan = _make_scan(**{"Dockerfile": "FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04\n"})
    hits = detect_nvidia_docker(scan)
    assert len(hits) == 1
    assert hits[0].id == "NVIDIA_DOCKER"


def test_detect_bitsandbytes():
    scan = _make_scan(**{"requirements.txt": "torch\nbitsandbytes\nvllm\n"})
    hits = detect_bitsandbytes(scan)
    assert len(hits) == 1
    assert hits[0].id == "BITSANDBYTES"


def test_detect_flash_attn():
    scan = _make_scan(**{"requirements.txt": "flash-attn\n"})
    hits = detect_flash_attn(scan)
    assert len(hits) == 1
    assert hits[0].id == "FLASH_ATTN"


def test_detect_xformers():
    scan = _make_scan(**{"requirements.txt": "xformers\n"})
    hits = detect_xformers(scan)
    assert len(hits) == 1
    assert hits[0].id == "XFORMERS"


def test_detect_triton():
    scan = _make_scan(**{"requirements.txt": "triton\n"})
    hits = detect_triton(scan)
    assert len(hits) == 1
    assert hits[0].id == "TRITON"


def test_no_false_positives_on_clean_file():
    scan = _make_scan(**{"clean.py": "import os\nprint('hello world')\n"})
    hits = detect_torch_cuda(scan)
    assert len(hits) == 0


def test_run_all_detectors_on_demo_repo():
    """run_all_detectors should find multiple issues in the demo repo."""
    scan = scan_repo(DEMO_REPO)
    detections = run_all_detectors(scan)
    detection_ids = {d.id for d in detections}
    assert "TORCH_CUDA" in detection_ids
    assert "TO_CUDA" in detection_ids
    assert "NVIDIA_DOCKER" in detection_ids
    assert "BITSANDBYTES" in detection_ids
    assert "FLASH_ATTN" in detection_ids
    assert "XFORMERS" in detection_ids
    assert "TRITON" in detection_ids


# ---------------------------------------------------------------------------
# Patcher tests
# ---------------------------------------------------------------------------

def test_patcher_generates_dockerfile_rocm():
    """generate_artifacts must create Dockerfile.rocm."""
    scan = scan_repo(DEMO_REPO)
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = generate_artifacts(scan, tmp)
        filenames = [a.filename for a in artifacts]
        assert "Dockerfile.rocm" in filenames
        docker_art = next(a for a in artifacts if a.filename == "Dockerfile.rocm")
        assert os.path.isfile(docker_art.output_path)


def test_patcher_generates_benchmark_rocm():
    """generate_artifacts must create benchmark_rocm.py."""
    scan = scan_repo(DEMO_REPO)
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = generate_artifacts(scan, tmp)
        filenames = [a.filename for a in artifacts]
        assert "benchmark_rocm.py" in filenames
        bench_art = next(a for a in artifacts if a.filename == "benchmark_rocm.py")
        assert os.path.isfile(bench_art.output_path)
        # Benchmark must not contain hardcoded numeric results
        assert "tokens_per_second = " not in bench_art.content
        assert "latency_ms = " not in bench_art.content
        # Must contain methodology markers
        assert (
            "placeholder" in bench_art.content.lower()
            or "methodology" in bench_art.content.lower()
        )


def test_patcher_generates_all_expected_artifacts():
    """All six expected artifacts must be generated."""
    expected = {
        "Dockerfile.rocm",
        "requirements-rocm.txt",
        "serve_vllm_rocm.py",
        "benchmark_rocm.py",
        "README_AMD_MIGRATION.md",
        "flightdeck.patch",
    }
    scan = scan_repo(DEMO_REPO)
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = generate_artifacts(scan, tmp)
        filenames = {a.filename for a in artifacts}
        assert expected == filenames


def test_patcher_does_not_overwrite_originals():
    """Patcher must write to the output directory, not the source repo."""
    scan = scan_repo(DEMO_REPO)
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = generate_artifacts(scan, tmp)
        for art in artifacts:
            assert art.output_path.startswith(tmp), (
                f"Artifact written outside tmp: {art.output_path}"
            )
