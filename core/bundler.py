"""Create downloadable ROCm FlightDeck migration bundles."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterable

EXPECTED_BUNDLE_FILES = [
    "Dockerfile.rocm",
    "requirements-rocm.txt",
    "serve_vllm_rocm.py",
    "benchmark_rocm.py",
    "README_AMD_MIGRATION.md",
    "migration_report.md",
    "pull_request_description.md",
    "flightdeck.patch",
]


def create_migration_bundle(
    repo_name: str,
    patch_dir: str,
    report_path: str,
    output_base: str,
    pr_description_path: str = "",
) -> str:
    """Create outputs/bundles/<repo_name>_migration_bundle.zip."""
    safe_name = re.sub(r"[^\w\-.]", "_", repo_name) or "repo"
    bundle_dir = Path(output_base).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / f"{safe_name}_migration_bundle.zip"

    patch_root = Path(patch_dir).resolve()
    report = Path(report_path).resolve()
    pr_description = Path(pr_description_path).resolve() if pr_description_path else None

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in _existing_bundle_files(patch_root, report, pr_description):
            zf.write(file_path, arcname=file_path.name)

    return str(bundle_path)


def _existing_bundle_files(patch_root: Path, report: Path, pr_description: Path | None) -> Iterable[Path]:
    for filename in EXPECTED_BUNDLE_FILES:
        if filename == "migration_report.md":
            if report.is_file():
                yield report
            continue
        if filename == "pull_request_description.md":
            if pr_description and pr_description.is_file():
                yield pr_description
            continue
        candidate = patch_root / filename
        if candidate.is_file():
            yield candidate
