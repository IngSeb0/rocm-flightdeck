"""Tests for deterministic compatibility detectors."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detectors import (
    detect_bitsandbytes,
    detect_cuda_call,
    detect_flash_attn,
    detect_missing_benchmark_rocm,
    detect_missing_dockerfile_rocm,
    detect_nvidia_docker,
    detect_to_cuda,
    detect_torch_cuda,
    detect_triton,
    detect_vllm,
    detect_xformers,
    run_all_detectors,
)
from core.models import ScanResult
from core.scanner import scan_repo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAD_DEMO = os.path.join(ROOT, "demo_repos", "nvidia_locked_vllm_demo")


def _make_scan(**files) -> ScanResult:
    scan = ScanResult(repo_path="/fake")
    for path, content in files.items():
        scan.files_scanned.append(path)
        scan.file_contents[path] = content
    return scan


def test_detect_torch_cuda():
    hits = detect_torch_cuda(_make_scan(**{"app.py": "import torch\ntorch.cuda.is_available()\n"}))
    assert [hit.id for hit in hits] == ["TORCH_CUDA"]


def test_device_abstraction_pattern_is_not_critical():
    scan = _make_scan(**{
        "app.py": 'import torch\ndevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")\nmodel.to(device)\n'
    })
    assert detect_torch_cuda(scan) == []
    assert detect_to_cuda(scan) == []


def test_detect_cuda_call():
    hits = detect_cuda_call(_make_scan(**{"model.py": "model.cuda()\n"}))
    assert [hit.id for hit in hits] == ["DOT_CUDA_CALL"]


def test_detect_to_cuda():
    scan = _make_scan(**{"train.py": 'model.to("cuda")\nother.to(\'cuda\')\n'})
    hits = detect_to_cuda(scan)
    assert hits
    assert hits[0].id == "TO_CUDA"


def test_detect_nvidia_docker():
    hits = detect_nvidia_docker(_make_scan(**{"Dockerfile": "FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04\n"}))
    assert [hit.id for hit in hits] == ["NVIDIA_DOCKER"]


def test_detect_dependencies():
    scan = _make_scan(**{"requirements.txt": "bitsandbytes\nflash-attn\nxformers\ntriton\nvllm\n"})
    assert detect_bitsandbytes(scan)[0].id == "BITSANDBYTES"
    assert detect_flash_attn(scan)[0].id == "FLASH_ATTN"
    assert detect_xformers(scan)[0].id == "XFORMERS"
    assert detect_triton(scan)[0].id == "TRITON"
    assert detect_vllm(scan)[0].id == "VLLM_PRESENT"


def test_legacy_files_are_not_active_blockers():
    scan = _make_scan(**{
        "legacy/benchmark.cuda.legacy.py": "import torch\nx = torch.randn(1).cuda()\n",
        "legacy/Dockerfile.nvidia.legacy": "FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04\n",
        "legacy/requirements.nvidia.legacy.txt": "bitsandbytes\nflash-attn\nxformers\ntriton\n",
        "Dockerfile": "FROM rocm/pytorch:latest\n",
        "requirements.txt": "torch\nvllm\n",
    })
    ids = {d.id for d in run_all_detectors(scan)}
    assert "DOT_CUDA_CALL" not in ids
    assert "NVIDIA_DOCKER" not in ids
    assert "BITSANDBYTES" not in ids
    assert "FLASH_ATTN" not in ids
    assert "XFORMERS" not in ids
    assert "TRITON" not in ids


def test_detect_missing_rocm_files():
    scan = _make_scan(**{"app.py": "print('hello')\n"})
    assert detect_missing_dockerfile_rocm(scan)[0].id == "MISSING_DOCKERFILE_ROCM"
    assert detect_missing_benchmark_rocm(scan)[0].id == "MISSING_BENCHMARK_ROCM"


def test_run_all_detectors_on_bad_demo_repo():
    scan = scan_repo(BAD_DEMO)
    ids = {d.id for d in run_all_detectors(scan)}
    expected = {
        "TORCH_CUDA",
        "DOT_CUDA_CALL",
        "TO_CUDA",
        "NVIDIA_DOCKER",
        "CUDA_HOME",
        "BITSANDBYTES",
        "FLASH_ATTN",
        "XFORMERS",
        "TRITON",
        "MISSING_DOCKERFILE_ROCM",
        "MISSING_BENCHMARK_ROCM",
        "VLLM_PRESENT",
    }
    assert expected.issubset(ids)
