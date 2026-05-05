"""Scanner: recursively reads safe text files from a repo directory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from core.models import ScanResult

ALLOWED_EXTENSIONS = {
    ".py",
    ".txt",
    ".yaml",
    ".yml",
    ".sh",
    ".md",
    ".toml",
}

ALLOWED_FILENAMES = {
    "Dockerfile",
    "Dockerfile.rocm",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
}

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "outputs",
}


def _is_safe_file(filename: str) -> bool:
    """Return True if this file should be scanned."""
    _, ext = os.path.splitext(filename)
    return ext in ALLOWED_EXTENSIONS or filename in ALLOWED_FILENAMES


def _is_text_content(raw: bytes) -> bool:
    """Return True if *raw* bytes decode as valid UTF-8."""
    try:
        raw.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def scan_repo(repo_path: str) -> ScanResult:
    """
    Recursively scan *repo_path* and return a ScanResult with file contents.

    Only files with allowed extensions / names are read.
    Binary files and ignored directories are skipped.
    Symlinks that escape the root directory are silently skipped.
    """
    result = ScanResult(repo_path=repo_path)

    # Canonicalise the root — resolves symlinks and normalises separators
    root = Path(repo_path).resolve()
    if not root.is_dir():
        return result

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place so os.walk won't descend into them
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]

        for filename in filenames:
            if not _is_safe_file(filename):
                continue

            candidate = Path(dirpath) / filename

            # Resolve symlinks; skip anything that escapes the root
            try:
                real = candidate.resolve()
                real.relative_to(root)  # raises ValueError if outside root
            except (OSError, ValueError):
                continue

            # Read raw bytes first so we can check encoding before opening as text
            try:
                raw = real.read_bytes()
            except OSError:
                continue

            # Limit probe to first 8 KB to keep memory usage predictable
            if not _is_text_content(raw[:8192]):
                continue

            rel_path = str(real.relative_to(root))
            try:
                content = raw.decode("utf-8", errors="replace")
                result.files_scanned.append(rel_path)
                result.file_contents[rel_path] = content
            except Exception:
                pass  # skip on any unexpected decode error

    return result
