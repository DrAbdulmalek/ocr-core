"""No-network model resolution tests for PaddleOCR and EasyOCR adapters.

All model files and provider modules are synthetic stubs. These tests never
install OCR engines, download weights, or process user documents.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from ocr_core.engines.easyocr import EasyOCREngine
from ocr_core.engines.paddle import PaddleEngine
from ocr_core.model_policy import ModelArtifactMissingError


@pytest.fixture(autouse=True)
def _clear_model_policy_environment(monkeypatch):
    monkeypatch.delenv("OCR_CORE_MODEL_POLICY", raising=False)
    monkeypatch.delenv("OCR_CORE_REMOTE_AUTHORIZED", raising=False)


def _fake_easyocr(monkeypatch):
    calls = []
    module = types.ModuleType("easyocr")

    def reader(**kwargs):
        calls.append(kwargs)
        return object()

    module.Reader = reader
    monkeypatch.setitem(sys.modules, "easyocr", module)
    return calls


def _fake_paddleocr(monkeypatch):
    calls = []
    module = types.ModuleType("paddleocr")

    class PaddleOCR:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    module.PaddleOCR = PaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    return calls


def _make_paddle_models(root: Path, *, include_rec: bool = True) -> None:
    for kind in ("det", "rec") if include_rec else ("det",):
        model_dir = root / kind
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "inference.pdmodel").write_bytes(b"synthetic-model")
        (model_dir / "inference.pdiparams").write_bytes(b"synthetic-weights")


def test_easyocr_disables_downloads_by_default(monkeypatch):
    calls = _fake_easyocr(monkeypatch)

    EasyOCREngine()._engine()

    assert len(calls) == 1
    assert calls[0]["download_enabled"] is False
    assert calls[0]["model_storage_directory"] is None


def test_easyocr_remote_download_requires_explicit_authorization(monkeypatch):
    calls = _fake_easyocr(monkeypatch)

    EasyOCREngine(
        policy="EXPLICIT_REMOTE",
        remote_authorized=False,
    )._engine()
    EasyOCREngine(
        policy="EXPLICIT_REMOTE",
        remote_authorized=True,
    )._engine()

    assert [call["download_enabled"] for call in calls] == [False, True]


def test_easyocr_remote_disabled_overrides_authorization(monkeypatch):
    calls = _fake_easyocr(monkeypatch)

    EasyOCREngine(
        policy="REMOTE_DISABLED",
        remote_authorized=True,
    )._engine()

    assert calls[0]["download_enabled"] is False


def test_easyocr_uses_existing_local_model_directory(monkeypatch, tmp_path):
    calls = _fake_easyocr(monkeypatch)
    model_dir = tmp_path / "easyocr-models"
    model_dir.mkdir()

    EasyOCREngine(model_storage_directory=model_dir)._engine()

    assert calls[0]["model_storage_directory"] == str(model_dir)
    assert calls[0]["download_enabled"] is False


def test_easyocr_rejects_missing_or_symlinked_model_directory(monkeypatch, tmp_path):
    calls = _fake_easyocr(monkeypatch)
    missing = tmp_path / "missing"
    with pytest.raises(ModelArtifactMissingError):
        EasyOCREngine(model_storage_directory=missing)._engine()
    assert calls == []

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ModelArtifactMissingError):
        EasyOCREngine(model_storage_directory=link)._engine()
    assert calls == []


def test_paddle_local_only_fails_before_provider_initialization(monkeypatch):
    calls = _fake_paddleocr(monkeypatch)
    monkeypatch.setattr(
        "ocr_core.engines.paddle._find_custom_model_dir",
        lambda: None,
    )

    with pytest.raises(ModelArtifactMissingError):
        PaddleEngine()._engine()

    assert calls == []


def test_paddle_passes_only_complete_local_detection_and_recognition_models(
    monkeypatch, tmp_path
):
    calls = _fake_paddleocr(monkeypatch)
    _make_paddle_models(tmp_path)

    PaddleEngine(model_dir=tmp_path)._engine()

    assert len(calls) == 1
    assert calls[0]["det_model_dir"] == str(tmp_path / "det")
    assert calls[0]["rec_model_dir"] == str(tmp_path / "rec")
    assert calls[0]["use_angle_cls"] is False


def test_paddle_partial_local_models_fail_closed_even_with_remote_opt_in(
    monkeypatch, tmp_path
):
    calls = _fake_paddleocr(monkeypatch)
    _make_paddle_models(tmp_path, include_rec=False)

    with pytest.raises(ModelArtifactMissingError):
        PaddleEngine(
            model_dir=tmp_path,
            policy="EXPLICIT_REMOTE",
            remote_authorized=True,
        )._engine()

    assert calls == []


def test_paddle_remote_resolution_requires_explicit_authorization(monkeypatch):
    calls = _fake_paddleocr(monkeypatch)
    monkeypatch.setattr(
        "ocr_core.engines.paddle._find_custom_model_dir",
        lambda: None,
    )

    with pytest.raises(ModelArtifactMissingError):
        PaddleEngine(
            policy="EXPLICIT_REMOTE",
            remote_authorized=False,
        )._engine()

    PaddleEngine(
        policy="EXPLICIT_REMOTE",
        remote_authorized=True,
    )._engine()

    assert len(calls) == 1
    assert "det_model_dir" not in calls[0]
    assert "rec_model_dir" not in calls[0]


def test_paddle_remote_disabled_overrides_authorization(monkeypatch):
    calls = _fake_paddleocr(monkeypatch)
    monkeypatch.setattr(
        "ocr_core.engines.paddle._find_custom_model_dir",
        lambda: None,
    )

    with pytest.raises(ModelArtifactMissingError):
        PaddleEngine(
            policy="REMOTE_DISABLED",
            remote_authorized=True,
        )._engine()

    assert calls == []
