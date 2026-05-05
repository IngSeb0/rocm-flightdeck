"""Tests for Hugging Face Space entrypoint behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_root_app_exposes_demo_without_running_analysis():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("flightdeck_space_app", app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert hasattr(module, "demo")
    assert module.demo is not None
