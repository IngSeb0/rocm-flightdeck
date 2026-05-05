"""Create a migrated ROCm-oriented repository copy and rescore it."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from core.detectors import run_all_detectors
from core.models import Detection, GeneratedArtifact, ScoreBreakdown
from core.scanner import scan_repo
from core.scoring import compute_score

IGNORED_COPY_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "node_modules",
    "outputs",
}

SAFE_COPY_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
}

SAFE_COPY_FILENAMES = {
    "Dockerfile",
    "Dockerfile.rocm",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
}

INCOMPATIBLE_DEPENDENCIES = {"bitsandbytes", "flash-attn", "xformers", "triton"}


@dataclass
class MigrationResult:
    before_score: ScoreBreakdown
    after_score: ScoreBreakdown
    improvement_points: int
    before_blockers: List[Detection]
    after_blockers: List[Detection]
    remaining_risks: List[str]
    migrated_repo_path: str
    after_detections: List[Detection] = field(default_factory=list)


def migrate_repo(source_repo_path: str, output_base: str, artifacts: List[GeneratedArtifact]) -> MigrationResult:
    """Create a migrated repository copy, apply safe transformations, and rescore it."""
    source = Path(source_repo_path).resolve()
    repo_name = re.sub(r"[^\w\-.]", "_", source.name) or "repo"
    output_root = Path(output_base).resolve()
    migrated = output_root / f"{repo_name}_rocm"

    before_scan = scan_repo(str(source))
    before_detections = run_all_detectors(before_scan)
    before_score = compute_score(before_detections)

    if migrated.exists():
        shutil.rmtree(migrated)
    migrated.mkdir(parents=True, exist_ok=True)

    _copy_safe_tree(source, migrated)
    _relocate_legacy_defaults(migrated)
    _transform_python_files(migrated)
    _add_generated_artifacts(migrated, artifacts)
    _promote_rocm_defaults(migrated)

    after_scan = scan_repo(str(migrated))
    after_detections = run_all_detectors(after_scan)
    after_score = compute_score(after_detections)

    remaining = _remaining_risks(after_detections)
    return MigrationResult(
        before_score=before_score,
        after_score=after_score,
        improvement_points=after_score.total_score - before_score.total_score,
        before_blockers=before_detections,
        after_blockers=after_detections,
        remaining_risks=remaining,
        migrated_repo_path=str(migrated),
        after_detections=after_detections,
    )


def _copy_safe_tree(source: Path, destination: Path) -> None:
    for root, dirnames, filenames in os.walk(source):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_COPY_DIRS]
        root_path = Path(root)
        rel_dir = root_path.relative_to(source)
        target_dir = destination / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            src = root_path / filename
            if not _is_safe_copy_file(src):
                continue
            target = target_dir / filename
            target.write_bytes(src.read_bytes())


def _is_safe_copy_file(path: Path) -> bool:
    return path.suffix in SAFE_COPY_EXTENSIONS or path.name in SAFE_COPY_FILENAMES


def _transform_python_files(repo_path: Path) -> None:
    for py_file in sorted(repo_path.rglob("*.py")):
        text = py_file.read_text(encoding="utf-8")
        updated = _transform_python_text(text)
        py_file.write_text(updated, encoding="utf-8")


def _transform_python_text(text: str) -> str:
    updated = text
    updated = updated.replace('DEVICE = torch.device("cuda")', 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")')
    updated = updated.replace("DEVICE = torch.device('cuda')", 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")')
    updated = updated.replace("DEVICE", "device")
    updated = updated.replace('.to("cuda")', ".to(device)")
    updated = updated.replace(".to('cuda')", ".to(device)")
    updated = updated.replace(".cuda()", ".to(device)")
    updated = updated.replace(
        'raise RuntimeError("CUDA is not available. This demo requires an NVIDIA GPU.")',
        'raise RuntimeError("GPU acceleration is not available. Install ROCm PyTorch on AMD or CUDA PyTorch on NVIDIA.")',
    )
    updated = updated.replace("if not torch.cuda.is_available():", "if device.type != \"cuda\":")
    updated = updated.replace(
        'print(f"CUDA available: {torch.cuda.is_available()}")',
        'print(f"GPU acceleration selected: {device.type == \'cuda\'}")',
    )
    if "import torch" in updated and "device = torch.device(" not in updated:
        updated = updated.replace(
            "import torch",
            'import torch\n\n# ROCm PyTorch exposes AMD GPUs through the CUDA-compatible PyTorch API.\n# Use a device abstraction for portability across AMD, NVIDIA, and CPU fallback.\ndevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
            1,
        )
    return updated


def _relocate_legacy_defaults(repo_path: Path) -> None:
    legacy_dir = repo_path / "legacy"
    dockerfile = repo_path / "Dockerfile"
    if dockerfile.exists() and _contains_any(dockerfile, {"nvidia/cuda", "CUDA_HOME"}):
        legacy_dir.mkdir(parents=True, exist_ok=True)
        dockerfile.replace(legacy_dir / "Dockerfile.nvidia.legacy")

    requirements = repo_path / "requirements.txt"
    if requirements.exists() and _requirements_have_nvidia_risks(requirements):
        legacy_dir.mkdir(parents=True, exist_ok=True)
        requirements.replace(legacy_dir / "requirements.nvidia.legacy.txt")

    benchmark = repo_path / "benchmark.py"
    if benchmark.exists() and _contains_any(benchmark, {".cuda()", "torch.cuda", '.to("cuda")', ".to('cuda')"}):
        legacy_dir.mkdir(parents=True, exist_ok=True)
        benchmark.replace(legacy_dir / "benchmark.cuda.legacy.py")


def _promote_rocm_defaults(repo_path: Path) -> None:
    docker_rocm = repo_path / "Dockerfile.rocm"
    if docker_rocm.exists():
        (repo_path / "Dockerfile").write_text(docker_rocm.read_text(encoding="utf-8"), encoding="utf-8")

    requirements_rocm = repo_path / "requirements-rocm.txt"
    if requirements_rocm.exists():
        (repo_path / "requirements.txt").write_text(requirements_rocm.read_text(encoding="utf-8"), encoding="utf-8")


def _contains_any(path: Path, needles: set[str]) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(needle in text for needle in needles)


def _requirements_have_nvidia_risks(path: Path) -> bool:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        package = stripped.split("==", 1)[0].split(">=", 1)[0].lower()
        if package in INCOMPATIBLE_DEPENDENCIES:
            return True
    return False


def _add_generated_artifacts(repo_path: Path, artifacts: List[GeneratedArtifact]) -> None:
    for artifact in artifacts:
        if artifact.filename == "flightdeck.patch":
            continue
        (repo_path / artifact.filename).write_text(artifact.content, encoding="utf-8")


def _remaining_risks(detections: List[Detection]) -> List[str]:
    risks = [
        f"[{d.severity}] {d.title}" for d in detections if d.severity in {"critical", "warning"}
    ]
    risks.append("Benchmarks not executed in this MVP; validate on AMD Developer Cloud / MI300X.")
    return risks
