"""Tests for the local ModelStore smoke-test helper."""
from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_test_model_store.py"
SPEC = importlib.util.spec_from_file_location("smoke_test_model_store", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_smoke_helper_forces_huggingface_offline_environment(monkeypatch):
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "DO_NOT_TRACK",
        "TOKENIZERS_PARALLELISM",
    ):
        monkeypatch.delenv(key, raising=False)

    MODULE._force_offline_environment()

    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"
    assert __import__("os").environ["TRANSFORMERS_OFFLINE"] == "1"
    assert __import__("os").environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert __import__("os").environ["DO_NOT_TRACK"] == "1"
    assert __import__("os").environ["TOKENIZERS_PARALLELISM"] == "false"


def test_peak_rss_is_reported_as_positive_bytes():
    assert MODULE._peak_rss_bytes() > 0
