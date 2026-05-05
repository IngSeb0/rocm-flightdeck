"""Test fixtures constrained to the writable sandbox paths."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

SANDBOX_TMP = Path(__file__).resolve().parents[1] / "outputs" / "test_tmp"


@pytest.fixture(scope="session", autouse=True)
def _ensure_sandbox_tempdir():
    SANDBOX_TMP.mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture
def tmp_path():
    SANDBOX_TMP.mkdir(parents=True, exist_ok=True)
    path = SANDBOX_TMP / f"test-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
