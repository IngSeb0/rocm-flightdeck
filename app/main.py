"""ROCm FlightDeck - Gradio Web Application."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from core.bundler import create_migration_bundle
from core.detectors import run_all_detectors
from core.migrator import migrate_repo
from core.models import FlightDeckResult
from core.patcher import generate_artifacts
from core.planner import generate_plan
from core.pr_writer import generate_pr_description
from core.reporter import generate_report
from core.scanner import scan_repo
from core.scoring import assessment_label, compute_score

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PATCHES_DIR = _PROJECT_ROOT / "outputs" / "patches"
_REPORTS_DIR = _PROJECT_ROOT / "outputs" / "reports"
_MIGRATED_DIR = _PROJECT_ROOT / "outputs" / "migrated_repos"
_BUNDLES_DIR = _PROJECT_ROOT / "outputs" / "bundles"

DEMO_REPOS = {
    "nvidia_locked_vllm_demo": "demo_repos/nvidia_locked_vllm_demo",
    "partially_portable_pytorch_demo": "demo_repos/partially_portable_pytorch_demo",
    "rocm_ready_vllm_demo": "demo_repos/rocm_ready_vllm_demo",
}


def run_analysis(demo_repo: str, repo_path: str):
    """Run scan -> plan -> patch -> migrate -> rescore -> report -> bundle."""
    selected_path = repo_path.strip() if repo_path and repo_path.strip() else DEMO_REPOS.get(demo_repo, "")
    if not selected_path:
        msg = "Please select a demo repo or provide a repository path."
        return (msg,) * 9 + (None,)

    abs_path = Path(selected_path)
    if not abs_path.is_absolute():
        abs_path = _PROJECT_ROOT / abs_path
    abs_path = abs_path.resolve()

    if not abs_path.is_dir():
        msg = f"Directory not found: `{abs_path}`"
        return (msg,) * 9 + (None,)

    scan = scan_repo(str(abs_path))
    detections = run_all_detectors(scan)
    before_score = compute_score(detections)
    plan = generate_plan(detections)
    artifacts = generate_artifacts(scan, str(_PATCHES_DIR))
    migration = migrate_repo(str(abs_path), str(_MIGRATED_DIR), artifacts)

    result = FlightDeckResult(
        repo_path=str(abs_path),
        scan=scan,
        detections=detections,
        score=before_score,
        plan=plan,
        artifacts=artifacts,
        before_score=before_score,
        after_score=migration.after_score,
        before_detections=detections,
        after_detections=migration.after_detections,
        improvement_points=migration.improvement_points,
        migrated_repo_path=migration.migrated_repo_path,
        remaining_risks=migration.remaining_risks,
    )

    report_path = generate_report(result, str(_REPORTS_DIR))
    result.report_path = report_path
    repo_name = abs_path.name
    patch_dir = _PATCHES_DIR / repo_name
    pr_path = generate_pr_description(result, str(_REPORTS_DIR))
    result.pr_description_path = pr_path
    report_path = generate_report(result, str(_REPORTS_DIR))
    result.report_path = report_path
    pr_path = generate_pr_description(result, str(_REPORTS_DIR))
    result.pr_description_path = pr_path
    bundle_path = create_migration_bundle(repo_name, str(patch_dir), report_path, str(_BUNDLES_DIR), pr_path)
    result.bundle_path = bundle_path
    report_path = generate_report(result, str(_REPORTS_DIR))
    result.report_path = report_path
    bundle_path = create_migration_bundle(repo_name, str(patch_dir), report_path, str(_BUNDLES_DIR), pr_path)
    result.bundle_path = bundle_path

    return (
        _format_overview(result),
        _format_before_after(result),
        _format_blockers(detections, "Before Migration Blockers"),
        _format_plan(plan),
        _load_artifact(artifacts, "flightdeck.patch"),
        _format_artifacts(artifacts),
        _load_report(report_path),
        _format_benchmark_status(),
        _format_bundle(result),
        bundle_path,
    )


def _format_overview(result: FlightDeckResult) -> str:
    critical = [d for d in result.detections if d.severity == "critical"]
    warnings = [d for d in result.detections if d.severity == "warning"]
    bundle_status = "Generated" if result.bundle_path else "Not generated"
    return "\n".join([
        "# ROCm FlightDeck",
        "",
        "| Card | Status |",
        "|------|--------|",
        f"| ROCm Readiness Score | {result.before_score.total_score}/100 -> {result.after_score.total_score}/100 |",
        f"| Assessment | {assessment_label(result.after_score.total_score)} |",
        f"| Critical Blockers | {len(critical)} |",
        f"| Warnings | {len(warnings)} |",
        f"| Generated Files | {len(result.artifacts)} artifacts + migration_report.md + pull_request_description.md |",
        f"| Migration Bundle Status | {bundle_status} |",
        "",
        "ROCm FlightDeck is not a generic CUDA-to-ROCm converter. It creates a measurable, reviewable migration workflow for vLLM / PyTorch inference repositories.",
    ])


def _format_before_after(result: FlightDeckResult) -> str:
    before_critical = [d for d in result.before_detections if d.severity == "critical"]
    remaining = [d for d in result.after_detections if d.severity in {"critical", "warning"}]
    return "\n".join([
        "## Before / After Migration",
        "",
        f"- Before ROCm Readiness Score: **{result.before_score.total_score}/100**",
        f"- Before assessment: **{assessment_label(result.before_score.total_score)}**",
        f"- After ROCm Readiness Score: **{result.after_score.total_score}/100**",
        f"- After assessment: **{assessment_label(result.after_score.total_score)}**",
        f"- Improvement points: **{result.improvement_points}**",
        f"- Before critical blockers: **{len(before_critical)}**",
        f"- Remaining blockers: **{len(remaining)}**",
        f"- Migrated repo path: `{result.migrated_repo_path}`",
    ])


def _format_blockers(detections, title: str) -> str:
    lines = [f"## {title}", ""]
    for severity in ("critical", "warning", "info"):
        group = [d for d in detections if d.severity == severity]
        lines.append(f"### {severity.title()} ({len(group)})")
        if not group:
            lines.append("_None._")
            lines.append("")
            continue
        for d in group:
            lines.append(f"**[{d.id}] {d.title}**")
            if d.file_path:
                lines.append(f"- File: `{d.file_path}`")
            if d.evidence:
                lines.append(f"- Evidence: `{d.evidence[:160]}`")
            lines.append(f"- Recommendation: {d.recommendation}")
            lines.append("")
    return "\n".join(lines)


def _format_plan(plan) -> str:
    lines = ["## Migration Plan", ""]
    for step in plan.steps:
        lines.append(f"- {step}")
    return "\n".join(lines)


def _format_artifacts(artifacts) -> str:
    lines = [f"## Generated Artifacts ({len(artifacts)} files)", ""]
    for artifact in artifacts:
        preview = "\n".join(artifact.content.splitlines()[:24])
        lines.extend([
            f"### `{artifact.filename}`",
            f"Written to: `{artifact.output_path}`",
            "",
            f"```text\n{preview}\n...\n```",
            "",
        ])
    return "\n".join(lines)


def _load_artifact(artifacts, filename: str) -> str:
    artifact = next((a for a in artifacts if a.filename == filename), None)
    if not artifact:
        return f"`{filename}` was not generated."
    return f"```diff\n{artifact.content}\n```"


def _load_report(report_path: str) -> str:
    try:
        return Path(report_path).read_text(encoding="utf-8")
    except OSError:
        return f"_Report not found at `{report_path}`_"


def _format_benchmark_status() -> str:
    return "\n".join([
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
    ])


def _format_bundle(result: FlightDeckResult) -> str:
    return "\n".join([
        "## Download Bundle",
        "",
        f"Bundle path: `{result.bundle_path}`",
        "",
        "Includes Dockerfile.rocm, requirements-rocm.txt, serve_vllm_rocm.py, benchmark_rocm.py, README_AMD_MIGRATION.md, migration_report.md, pull_request_description.md, and flightdeck.patch.",
    ])


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ROCm FlightDeck") as demo:
        gr.Markdown(
            "# ROCm FlightDeck\n"
            "**AI performance portability lab for AMD ROCm / MI300X.**\n\n"
            "Detect -> Plan -> Patch -> Create migrated repo -> Re-score -> Report -> Download bundle."
        )

        with gr.Row():
            demo_input = gr.Dropdown(
                choices=list(DEMO_REPOS.keys()),
                value="nvidia_locked_vllm_demo",
                label="Demo Repository",
                scale=2,
            )
            repo_input = gr.Textbox(
                label="Manual Repository Path",
                placeholder="Leave blank to use selected demo repo",
                scale=3,
            )
            run_btn = gr.Button("Run Analysis", variant="primary", scale=1)

        with gr.Tabs():
            with gr.Tab("Overview"):
                overview_out = gr.Markdown()
            with gr.Tab("Before / After"):
                before_after_out = gr.Markdown()
            with gr.Tab("Blockers"):
                blockers_out = gr.Markdown()
            with gr.Tab("Migration Plan"):
                plan_out = gr.Markdown()
            with gr.Tab("Generated Patch"):
                patch_out = gr.Markdown()
            with gr.Tab("Artifacts"):
                artifacts_out = gr.Markdown()
            with gr.Tab("Technical Report"):
                report_out = gr.Markdown()
            with gr.Tab("Benchmark Status"):
                benchmark_out = gr.Markdown()
            with gr.Tab("Download Bundle"):
                bundle_md_out = gr.Markdown()
                bundle_file_out = gr.File(label="Migration Bundle")

        run_btn.click(
            fn=run_analysis,
            inputs=[demo_input, repo_input],
            outputs=[
                overview_out,
                before_after_out,
                blockers_out,
                plan_out,
                patch_out,
                artifacts_out,
                report_out,
                benchmark_out,
                bundle_md_out,
                bundle_file_out,
            ],
        )
    return demo


demo = build_ui()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
