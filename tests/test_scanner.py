"""Tests for core/scanner.py."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scanner import scan_repo

DEMO_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "demo_repos",
    "nvidia_locked_vllm_demo",
)


def test_scanner_finds_files_in_demo_repo():
    """Scanner must find at least the expected files in the demo repo."""
    result = scan_repo(DEMO_REPO)
    assert len(result.files_scanned) > 0, "Expected at least one scanned file"
    file_names = [os.path.basename(p) for p in result.files_scanned]
    assert "app.py" in file_names
    assert "requirements.txt" in file_names
    assert "Dockerfile" in file_names


def test_scanner_returns_file_contents():
    """Scanner must return non-empty file contents."""
    result = scan_repo(DEMO_REPO)
    assert len(result.file_contents) > 0
    for path, content in result.file_contents.items():
        assert isinstance(content, str), f"Content for {path} should be a string"
        assert len(content) > 0, f"Content for {path} should not be empty"


def test_scanner_ignores_git_dir():
    """Scanner must not descend into .git directories."""
    result = scan_repo(DEMO_REPO)
    for path in result.files_scanned:
        assert ".git" not in path.split(os.sep), f"Should not scan .git: {path}"


def test_scanner_ignores_binary_files():
    """Scanner skips files that are not valid UTF-8 text."""
    with tempfile.TemporaryDirectory() as tmp:
        binary_file = os.path.join(tmp, "binary.py")
        with open(binary_file, "wb") as fh:
            fh.write(b"\xff\xfe\x00\x01" * 100)
        result = scan_repo(tmp)
        assert "binary.py" not in result.files_scanned


def test_scanner_only_scans_allowed_extensions():
    """Scanner must not include disallowed extensions."""
    result = scan_repo(DEMO_REPO)
    for path in result.files_scanned:
        _, ext = os.path.splitext(path)
        filename = os.path.basename(path)
        allowed_ext = {".py", ".txt", ".yaml", ".yml", ".sh", ".md", ".toml"}
        allowed_names = {"Dockerfile", "Dockerfile.rocm", "requirements.txt"}
        assert ext in allowed_ext or filename in allowed_names, (
            f"Unexpected file scanned: {path}"
        )


def test_scanner_repo_path_preserved():
    """ScanResult.repo_path must match the input."""
    result = scan_repo(DEMO_REPO)
    assert result.repo_path == DEMO_REPO
