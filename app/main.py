"""
ROCm FlightDeck — Gradio Web Application.

Run with:
    python app/main.py
"""

from __future__ import annotations

import os
import sys

# Allow imports from the project root regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from core.detectors import run_all_detectors
from core.models import FlightDeckResult
from core.patcher import generate_artifacts
from core.planner import generate_plan
from core.reporter import generate_report
from core.scanner import scan_repo
from core.scoring import compute_score

# ---------------------------------------------------------------------------
# Output directories (relative to project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATCHES_DIR = os.path.join(_PROJECT_ROOT, "outputs", "patches")
_REPORTS_DIR = os.path.join(_PROJECT_ROOT, "outputs", "reports")


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def run_analysis(repo_path: str):
    """
    Full FlightDeck analysis pipeline.
    Returns six strings: score_md, blockers_md, plan_md, artifacts_md, report_md, status.
    """
    if not repo_path or not repo_path.strip():
        msg = "⚠️ Please provide a repository path."
        return msg, msg, msg, msg, msg, msg

    # Resolve relative paths against the project root, then canonicalize
    abs_path = repo_path.strip()
    if not os.path.isabs(abs_path):
        abs_path = os.path.join(_PROJECT_ROOT, abs_path)
    abs_path = os.path.realpath(abs_path)

    if not os.path.isdir(abs_path):
        msg = f"❌ Directory not found: `{abs_path}`"
        return msg, msg, msg, msg, msg, msg

    # 1. Scan
    scan = scan_repo(abs_path)

    # 2. Detect
    detections = run_all_detectors(scan)

    # 3. Score
    score = compute_score(detections)

    # 4. Plan
    plan = generate_plan(detections)

    # 5. Patch
    artifacts = generate_artifacts(scan, _PATCHES_DIR)

    # 6. Build result object
    result = FlightDeckResult(
        repo_path=abs_path,
        scan=scan,
        detections=detections,
        score=score,
        plan=plan,
        artifacts=artifacts,
    )

    # 7. Report
    report_path = generate_report(result, _REPORTS_DIR)
    result.report_path = report_path

    # --- Format outputs ---
    score_md = _format_score(score)
    blockers_md = _format_blockers(detections)
    plan_md = _format_plan(plan)
    artifacts_md = _format_artifacts(artifacts)
    report_md = _load_report(report_path)
    status = f"✅ Analysis complete. Scanned **{len(scan.files_scanned)}** files."

    return score_md, blockers_md, plan_md, artifacts_md, report_md, status


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_score(score) -> str:
    total = score.total_score
    bar = "🟢" if total >= 70 else ("🟡" if total >= 40 else "🔴")
    lines = [
        f"## {bar} ROCm Readiness Score: **{total} / 100**",
        "",
        "| Category | Score | Max |",
        "|----------|-------|-----|",
        f"| Device abstraction | {score.device_abstraction} | 20 |",
        f"| Dependency compatibility | {score.dependency_compatibility} | 25 |",
        f"| Docker / runtime | {score.docker_runtime} | 20 |",
        f"| vLLM serving readiness | {score.vllm_serving} | 20 |",
        f"| Benchmark readiness | {score.benchmark_readiness} | 15 |",
        "",
        "### Assessment",
        score.explanation,
        "",
        "### Deductions",
    ]
    for d in score.deductions:
        lines.append(f"- {d}")
    return "\n".join(lines)


def _format_blockers(detections) -> str:
    critical = [d for d in detections if d.severity == "critical"]
    warnings = [d for d in detections if d.severity == "warning"]
    infos = [d for d in detections if d.severity == "info"]

    lines = [f"## Detections ({len(detections)} total)", ""]

    if critical:
        lines.append("### ❌ Critical Blockers")
        for d in critical:
            lines.append(f"**[{d.id}] {d.title}**")
            if d.file_path:
                lines.append(f"- File: `{d.file_path}`")
            if d.evidence:
                lines.append(f"- Evidence: `{d.evidence[:120]}`")
            lines.append(f"- 💡 {d.recommendation}")
            lines.append("")

    if warnings:
        lines.append("### ⚠️ Warnings")
        for d in warnings:
            lines.append(f"**[{d.id}] {d.title}**")
            if d.file_path:
                lines.append(f"- File: `{d.file_path}`")
            lines.append(f"- 💡 {d.recommendation}")
            lines.append("")

    if infos:
        lines.append("### ℹ️ Info")
        for d in infos:
            lines.append(f"**[{d.id}] {d.title}**")
            lines.append(f"- 💡 {d.recommendation}")
            lines.append("")

    return "\n".join(lines)


def _format_plan(plan) -> str:
    lines = ["## Migration Plan", ""]
    for step in plan.steps:
        lines.append(f"- {step}")
        lines.append("")
    return "\n".join(lines)


def _format_artifacts(artifacts) -> str:
    lines = [f"## Generated Artifacts ({len(artifacts)} files)", ""]
    for art in artifacts:
        lines.append(f"### 📄 `{art.filename}`")
        lines.append(f"**Written to:** `{art.output_path}`")
        lines.append("")
        # Show first 30 lines of each artifact
        preview = "\n".join(art.content.splitlines()[:30])
        lines.append(f"```\n{preview}\n...\n```")
        lines.append("")
    return "\n".join(lines)


def _load_report(report_path: str) -> str:
    try:
        with open(report_path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return f"_Report not found at `{report_path}`_"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ROCm FlightDeck") as demo:
        gr.Markdown(
            "# 🚀 ROCm FlightDeck\n"
            "**Autonomous performance portability for LLM inference on AMD GPUs.**\n\n"
            "Analyze a CUDA/NVIDIA-first repository and generate a ROCm migration plan, "
            "readiness score, and starter artifacts — all with deterministic static analysis, "
            "no internet required."
        )

        with gr.Row():
            repo_input = gr.Textbox(
                label="Repository Path",
                value="demo_repos/nvidia_locked_vllm_demo",
                placeholder="Path to local repository (relative or absolute)",
                scale=4,
            )
            run_btn = gr.Button("🔍 Run Analysis", variant="primary", scale=1)

        status_box = gr.Markdown("_Enter a repo path and click Run Analysis._")

        with gr.Tabs():
            with gr.Tab("📊 Readiness Score"):
                score_out = gr.Markdown()
            with gr.Tab("🚫 Blockers"):
                blockers_out = gr.Markdown()
            with gr.Tab("🗺️ Migration Plan"):
                plan_out = gr.Markdown()
            with gr.Tab("📁 Generated Artifacts"):
                artifacts_out = gr.Markdown()
            with gr.Tab("📋 Technical Report"):
                report_out = gr.Markdown()

        run_btn.click(
            fn=run_analysis,
            inputs=[repo_input],
            outputs=[score_out, blockers_out, plan_out, artifacts_out, report_out, status_box],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
