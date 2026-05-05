"""Reporter: generates the migration_report.md file."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from core.models import FlightDeckResult


def generate_report(result: FlightDeckResult, output_base: str) -> str:
    """
    Write migration_report.md to `output_base/<repo_name>/migration_report.md`.
    Returns the path to the written file.
    """
    repo_name = Path(result.repo_path).resolve().name
    # Sanitize to prevent path traversal in the output directory name
    repo_name = re.sub(r"[^\w\-.]", "_", repo_name) or "repo"
    out_dir = Path(output_base).resolve() / repo_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "migration_report.md"

    lines = _build_report(result, repo_name)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    return str(out_path)


def _build_report(result: FlightDeckResult, repo_name: str) -> List[str]:
    score = result.score
    detections = result.detections
    plan = result.plan
    artifacts = result.artifacts

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    critical = [d for d in detections if d.severity == "critical"]
    warnings = [d for d in detections if d.severity == "warning"]
    infos = [d for d in detections if d.severity == "info"]

    lines = [
        "# ROCm FlightDeck — Migration Report",
        f"**Repository:** `{repo_name}`  ",
        f"**Generated:** {now}  ",
        "**Tool:** ROCm FlightDeck MVP  ",
        "",
        "---",
        "",
        "## Summary",
        "",
        (
            f"ROCm FlightDeck analysed `{repo_name}` and found "
            f"**{len(critical)} critical blocker(s)**, "
            f"**{len(warnings)} warning(s)**, and "
            f"**{len(infos)} informational item(s)**."
        ),
        "",
        "The repository requires migration work before it can run reliably on AMD ROCm / MI300X.",
        "",
        "---",
        "",
        "## ROCm Readiness Score",
        "",
        f"### **{score.total_score} / 100**",
        "",
        "| Category | Score | Max |",
        "|----------|-------|-----|",
        f"| Device abstraction | {score.device_abstraction} | 20 |",
        f"| Dependency compatibility | {score.dependency_compatibility} | 25 |",
        f"| Docker / runtime | {score.docker_runtime} | 20 |",
        f"| vLLM serving readiness | {score.vllm_serving} | 20 |",
        f"| Benchmark readiness | {score.benchmark_readiness} | 15 |",
        "",
        f"**Assessment:** {score.explanation}",
        "",
        "### Deductions",
        "",
    ]

    if score.deductions:
        for d in score.deductions:
            lines.append(f"- {d}")
    else:
        lines.append("- No deductions.")

    lines += [
        "",
        "---",
        "",
        "## Critical Blockers",
        "",
    ]

    if critical:
        for d in critical:
            lines.append(f"### ❌ [{d.id}] {d.title}")
            if d.file_path:
                lines.append(f"**File:** `{d.file_path}`  ")
            if d.evidence:
                lines.append("**Evidence:**")
                lines.append("```")
                lines.append(d.evidence)
                lines.append("```")
            lines.append(f"**Recommendation:** {d.recommendation}")
            lines.append("")
    else:
        lines.append("_No critical blockers detected._")
        lines.append("")

    lines += [
        "---",
        "",
        "## Warnings",
        "",
    ]

    if warnings:
        for d in warnings:
            lines.append(f"### ⚠️ [{d.id}] {d.title}")
            if d.file_path:
                lines.append(f"**File:** `{d.file_path}`  ")
            if d.evidence:
                lines.append("**Evidence:**")
                lines.append("```")
                lines.append(d.evidence)
                lines.append("```")
            lines.append(f"**Recommendation:** {d.recommendation}")
            lines.append("")
    else:
        lines.append("_No warnings._")
        lines.append("")

    lines += [
        "---",
        "",
        "## Migration Plan",
        "",
    ]

    for step in plan.steps:
        lines.append(f"- {step}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Generated Files",
        "",
        "The following files were generated in `outputs/patches/`:",
        "",
    ]

    for art in artifacts:
        lines.append(f"- `{art.filename}` → `{art.output_path}`")

    lines += [
        "",
        "---",
        "",
        "## Benchmark Status",
        "",
        "> **Benchmark status: not executed in this local MVP.**  ",
        "> Run `benchmark_rocm.py` on AMD Developer Cloud / MI300X for real results.",
        "",
        "---",
        "",
        "## Remaining Risks",
        "",
        "- bitsandbytes quantization may not be fully supported on ROCm without a community fork.",
        "- Triton kernels need individual validation on the HIP backend.",
        "- flash-attn must be replaced with a ROCm-compatible implementation.",
        "- Driver and ROCm version compatibility must be validated end-to-end.",
        "- Performance parity with NVIDIA is not guaranteed and requires real benchmarking.",
        "",
        "---",
        "",
        "## Next Steps",
        "",
        "1. Address all **critical** blockers listed above.",
        "2. Review and apply `flightdeck.patch` (in outputs/patches/).",
        "3. Build and test `Dockerfile.rocm` on an AMD machine.",
        "4. Install ROCm PyTorch wheel and `requirements-rocm.txt`.",
        "5. Run `serve_vllm_rocm.py` and validate the serving endpoint.",
        "6. Run `benchmark_rocm.py` on MI300X and record real performance numbers.",
        "7. Fill in the benchmark table in `README_AMD_MIGRATION.md`.",
        "",
        "---",
        "",
        "_Generated by ROCm FlightDeck — open-source AI performance portability agent for AMD GPUs._",
    ]

    return lines
