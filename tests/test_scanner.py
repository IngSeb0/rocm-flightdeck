"""Tests for core/scanner.py."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scanner import scan_repo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_ROOT = os.path.join(ROOT, "demo_repos")


def _demo(name: str) -> str:
    return os.path.join(DEMO_ROOT, name)


def test_scanner_finds_expected_files_in_each_demo_repo():
    expected = {
        "nvidia_locked_vllm_demo": {"app.py", "requirements.txt", "Dockerfile"},
        "partially_portable_pytorch_demo": {"app.py", "requirements.txt", "Dockerfile"},
        "rocm_ready_vllm_demo": {"app.py", "requirements-rocm.txt", "Dockerfile.rocm", "benchmark_rocm.py"},
    }
    for repo_name, filenames in expected.items():
        result = scan_repo(_demo(repo_name))
        scanned = {os.path.basename(path) for path in result.files_scanned}
        assert filenames.issubset(scanned)


def test_scanner_returns_file_contents():
    result = scan_repo(_demo("nvidia_locked_vllm_demo"))
    assert result.file_contents
    for content in result.file_contents.values():
        assert isinstance(content, str)
        assert content


def test_scanner_ignores_unsafe_or_irrelevant_folders(tmp_path):
    (tmp_path / "app.py").write_text("print('scan me')\n", encoding="utf-8")
    for folder in [".git", "__pycache__", "node_modules", ".venv", "outputs"]:
        ignored = tmp_path / folder
        ignored.mkdir()
        (ignored / "hidden.py").write_text("torch.cuda\n", encoding="utf-8")

    result = scan_repo(str(tmp_path))

    assert "app.py" in result.files_scanned
    assert all("hidden.py" not in path for path in result.files_scanned)


def test_scanner_ignores_binary_files(tmp_path):
    (tmp_path / "binary.py").write_bytes(b"\xff\xfe\x00\x01" * 100)
    result = scan_repo(str(tmp_path))
    assert "binary.py" not in result.files_scanned


def test_scanner_only_scans_allowed_extensions():
    result = scan_repo(_demo("nvidia_locked_vllm_demo"))
    for path in result.files_scanned:
        _, ext = os.path.splitext(path)
        filename = os.path.basename(path)
        allowed_ext = {".py", ".txt", ".yaml", ".yml", ".sh", ".md", ".toml"}
        allowed_names = {"Dockerfile", "Dockerfile.rocm", "requirements.txt", "pyproject.toml", "README.md"}
        assert ext in allowed_ext or filename in allowed_names
