"""Reporter: generates the migration_report.md file."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from core.models import Detection, FlightDeckResult, ScoreBreakdown
from core.scoring import assessment_label


def generate_report(result: FlightDeckResult, output_base: str) -> str:
    """Write migration_report.md to `output_base/<repo_name>/migration_report.md`."""
    repo_name = Path(result.repo_path).resolve().name
    safe_name = re.sub(r"[^\w\-.]", "_", repo_name) or "repo"
    out_dir = Path(output_base).resolve() / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "migration_report.md"

    out_path.write_text("\n".join(_build_report(result, safe_name)), encoding="utf-8")
    return str(out_path)


def _build_report(result: FlightDeckResult, repo_name: str) -> List[str]:
    before_score = result.before_score or result.score
    after_score = result.after_score or result.score
    before_detections = result.before_detections or result.detections
    after_detections = result.after_detections or result.detections
    critical = [d for d in before_detections if d.severity == "critical"]
    warnings = [d for d in before_detections if d.severity == "warning"]
    remaining = [d for d in after_detections if d.severity in {"critical", "warning"}]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# ROCm FlightDeck - Migration Report",
        f"**Repository:** `{repo_name}`  ",
        f"**Generated:** {now}  ",
        "**Tool:** ROCm FlightDeck MVP  ",
        "",
        "## Executive Summary",
        "",
        (
            f"ROCm FlightDeck analyzed `{repo_name}`, generated reviewable ROCm starter artifacts, "
            "created a migrated repository copy, and rescored that copy using deterministic static analysis."
        ),
        "",
        f"- Before Score: **{before_score.total_score}/100**",
        f"- Before Assessment: **{assessment_label(before_score.total_score)}**",
        f"- After Score: **{after_score.total_score}/100**",
        f"- After Assessment: **{assessment_label(after_score.total_score)}**",
        f"- Score Improvement: **{result.improvement_points} point(s)**",
        f"- Critical Blockers: **{len(critical)}**",
        f"- Warnings: **{len(warnings)}**",
        "",
        "Benchmarks not executed in this MVP.",
        "",
        "## Before Score",
        "",
        *_score_lines(before_score),
        "",
        "## After Score",
        "",
        *_score_lines(after_score),
        "",
        "## Score Improvement",
        "",
        f"The migrated copy improved by **{result.improvement_points} point(s)**.",
        "",
        "## Category Breakdown",
        "",
        "| Category | Before | After | Max |",
        "|----------|--------|-------|-----|",
        f"| Device abstraction | {before_score.device_abstraction} | {after_score.device_abstraction} | 20 |",
        f"| Dependency compatibility | {before_score.dependency_compatibility} | {after_score.dependency_compatibility} | 25 |",
        f"| Docker / runtime | {before_score.docker_runtime} | {after_score.docker_runtime} | 20 |",
        f"| vLLM serving readiness | {before_score.vllm_serving} | {after_score.vllm_serving} | 20 |",
        f"| Benchmark readiness | {before_score.benchmark_readiness} | {after_score.benchmark_readiness} | 15 |",
        "",
        "## Critical Blockers",
        "",
        *_detection_lines(critical, "_No critical blockers detected._"),
        "",
        "## Warnings",
        "",
        *_detection_lines(warnings, "_No warnings detected._"),
        "",
        "## Migration Plan",
        "",
    ]

    for step in result.plan.steps:
        lines.append(f"- {step}")

    lines.extend([
        "",
        "## Generated Artifacts",
        "",
    ])
    for artifact in result.artifacts:
        lines.append(f"- `{artifact.filename}` -> `{artifact.output_path}`")

    lines.extend([
        "",
        "## Migrated Repo Path",
        "",
        f"`{result.migrated_repo_path or 'Not generated'}`",
        "",
        "## Bundle Path",
        "",
        f"`{result.bundle_path or 'Not generated'}`",
        "",
        "## Pull Request Description",
        "",
        f"`{result.pr_description_path or 'Not generated'}`",
        "",
        "## Generated Patch",
        "",
        "`flightdeck.patch` is included in the generated artifacts and migration bundle. It contains reviewable starter hunks and must be validated before use.",
        "",
        "## Benchmark Status",
        "",
        "Benchmark status: not executed locally.",
        "",
        "Reason: This MVP does not run GPU workloads on the user machine.",
        "",
        "Next step: Run `benchmark_rocm.py` on AMD Developer Cloud / MI300X.",
        "",
        "Metrics to collect:",
        "- tokens/sec",
        "- latency",
        "- VRAM usage",
        "- cold start time",
        "- test pass rate",
        "",
        "## Remaining Risks",
        "",
    ])

    for risk in (result.remaining_risks or [f"[{d.severity}] {d.title}" for d in remaining]):
        lines.append(f"- {risk}")

    lines.extend([
        "",
        "## Next Steps",
        "",
        "1. Review every generated artifact and `flightdeck.patch` hunk.",
        "2. Build `Dockerfile.rocm` on an AMD ROCm machine.",
        "3. Install ROCm PyTorch and ROCm-compatible vLLM dependencies.",
        "4. Run functional tests on AMD Developer Cloud / MI300X.",
        "5. Execute `benchmark_rocm.py` on real MI300X hardware and record real metrics.",
        "",
        "## Build in Public Summary",
        "",
        (
            f"Today we migrated a CUDA-first vLLM inference demo repo toward AMD ROCm. "
            f"Before: ROCm Readiness Score {before_score.total_score}/100. "
            f"After: ROCm Readiness Score {after_score.total_score}/100. "
            "Generated: Dockerfile.rocm, requirements-rocm.txt, serve_vllm_rocm.py, "
            "benchmark_rocm.py, README_AMD_MIGRATION.md and flightdeck.patch. "
            "Benchmark status: pending real execution on AMD Developer Cloud / MI300X."
        ),
        "",
        "_Generated by ROCm FlightDeck - open-source AI performance portability agent for AMD GPUs._",
    ])
    return lines


def _score_lines(score: ScoreBreakdown) -> List[str]:
    lines = [
        f"### {score.total_score} / 100",
        "",
        f"**Assessment:** {assessment_label(score.total_score)}",
        "",
        "| Category | Score | Max |",
        "|----------|-------|-----|",
        f"| Device abstraction | {score.device_abstraction} | 20 |",
        f"| Dependency compatibility | {score.dependency_compatibility} | 25 |",
        f"| Docker / runtime | {score.docker_runtime} | 20 |",
        f"| vLLM serving readiness | {score.vllm_serving} | 20 |",
        f"| Benchmark readiness | {score.benchmark_readiness} | 15 |",
        "",
        score.explanation,
        "",
        "Deductions:",
    ]
    if not score.deductions:
        lines.append("- No deductions.")
    for deduction in score.deductions:
        file_text = f" in `{deduction['file']}`" if deduction.get("file") else ""
        lines.append(
            f"- -{deduction['points']} {deduction['category']}: "
            f"{deduction['reason']} ({deduction['severity']}){file_text}. "
            f"Recommendation: {deduction['recommendation']}"
        )
    return lines


def _detection_lines(detections: List[Detection], empty: str) -> List[str]:
    if not detections:
        return [empty]
    lines: List[str] = []
    for detection in detections:
        lines.append(f"### [{detection.id}] {detection.title}")
        if detection.file_path:
            lines.append(f"**File:** `{detection.file_path}`  ")
        if detection.evidence:
            lines.append("**Evidence:**")
            lines.append("```")
            lines.append(detection.evidence)
            lines.append("```")
        lines.append(f"**Recommendation:** {detection.recommendation}")
        lines.append("")
    return lines
