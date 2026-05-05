"""Tests for migration bundle generation."""

from __future__ import annotations

import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bundler import EXPECTED_BUNDLE_FILES, create_migration_bundle
from core.models import FlightDeckResult
from core.patcher import generate_artifacts
from core.planner import generate_plan
from core.reporter import generate_report
from core.pr_writer import generate_pr_description
from core.scanner import scan_repo
from core.detectors import run_all_detectors
from core.scoring import compute_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAD_DEMO = os.path.join(ROOT, "demo_repos", "nvidia_locked_vllm_demo")


def test_bundle_zip_contains_expected_files(tmp_path):
    scan = scan_repo(BAD_DEMO)
    detections = run_all_detectors(scan)
    score = compute_score(detections)
    artifacts = generate_artifacts(scan, str(tmp_path / "patches"))
    result = FlightDeckResult(
        repo_path=BAD_DEMO,
        scan=scan,
        detections=detections,
        score=score,
        plan=generate_plan(detections),
        artifacts=artifacts,
        before_score=score,
        after_score=score,
    )
    report = generate_report(result, str(tmp_path / "reports"))
    pr_description = generate_pr_description(result, str(tmp_path / "reports"))

    bundle = create_migration_bundle(
        "nvidia_locked_vllm_demo",
        str(tmp_path / "patches" / "nvidia_locked_vllm_demo"),
        report,
        str(tmp_path / "bundles"),
        pr_description,
    )

    assert os.path.isfile(bundle)
    with zipfile.ZipFile(bundle) as zf:
        assert set(EXPECTED_BUNDLE_FILES).issubset(set(zf.namelist()))
