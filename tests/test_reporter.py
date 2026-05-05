"""Tests for migration report generation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detectors import run_all_detectors
from core.migrator import migrate_repo
from core.models import FlightDeckResult
from core.patcher import generate_artifacts
from core.planner import generate_plan
from core.pr_writer import generate_pr_description
from core.reporter import generate_report
from core.scanner import scan_repo
from core.scoring import compute_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAD_DEMO = os.path.join(ROOT, "demo_repos", "nvidia_locked_vllm_demo")


def test_report_includes_migration_fields(tmp_path):
    scan = scan_repo(BAD_DEMO)
    detections = run_all_detectors(scan)
    score = compute_score(detections)
    artifacts = generate_artifacts(scan, str(tmp_path / "patches"))
    migration = migrate_repo(BAD_DEMO, str(tmp_path / "migrated"), artifacts)
    result = FlightDeckResult(
        repo_path=BAD_DEMO,
        scan=scan,
        detections=detections,
        score=score,
        plan=generate_plan(detections),
        artifacts=artifacts,
        before_score=score,
        after_score=migration.after_score,
        before_detections=detections,
        after_detections=migration.after_detections,
        improvement_points=migration.improvement_points,
        migrated_repo_path=migration.migrated_repo_path,
        bundle_path=str(tmp_path / "bundle.zip"),
        remaining_risks=migration.remaining_risks,
    )

    report_path = generate_report(result, str(tmp_path / "reports"))
    report = open(report_path, encoding="utf-8").read()

    assert os.path.isfile(report_path)
    assert "Before Score" in report
    assert "After Score" in report
    assert "Generated Artifacts" in report
    assert "Benchmark status: not executed locally." in report
    assert "Remaining Risks" in report
    assert "ROCm Candidate" in report or "ROCm Ready" in report


def test_pr_description_is_generated(tmp_path):
    scan = scan_repo(BAD_DEMO)
    detections = run_all_detectors(scan)
    score = compute_score(detections)
    artifacts = generate_artifacts(scan, str(tmp_path / "patches"))
    migration = migrate_repo(BAD_DEMO, str(tmp_path / "migrated"), artifacts)
    result = FlightDeckResult(
        repo_path=BAD_DEMO,
        scan=scan,
        detections=detections,
        score=score,
        plan=generate_plan(detections),
        artifacts=artifacts,
        before_score=score,
        after_score=migration.after_score,
        before_detections=detections,
        after_detections=migration.after_detections,
        improvement_points=migration.improvement_points,
        migrated_repo_path=migration.migrated_repo_path,
        remaining_risks=migration.remaining_risks,
    )

    pr_path = generate_pr_description(result, str(tmp_path / "reports"))
    pr = open(pr_path, encoding="utf-8").read()

    assert os.path.isfile(pr_path)
    assert "PR: Add ROCm migration starter artifacts" in pr
    assert "Before / After ROCm Readiness Score" in pr
    assert "Benchmark Status" in pr
    assert "Validation Checklist" in pr
