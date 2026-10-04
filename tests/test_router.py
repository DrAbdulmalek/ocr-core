"""Tests for ocr_core.engines.router (ported from omni engine_router)."""
import pytest

import ocr_core.engines.router as router_module
from ocr_core.engines.router import (
    ENGINE_EASYOCR,
    ENGINE_NOUGAT,
    ENGINE_PADDLE,
    ENGINE_QARI,
    ENGINE_QWEN_HANDWRITTEN,
    ENGINE_TESSERACT,
    ENGINE_TROCR,
    EngineRouter,
)


def test_low_profile_single_engine():
    r = EngineRouter(profile="low")
    chosen, reasons = r.select()
    assert chosen == [ENGINE_TESSERACT]
    assert reasons == ["low-end profile — single engine mode"]


def test_handwriting_block_prefers_qwen_then_qari():
    # generous RAM budget: Qwen(5.5GB)+QARI(4.5GB)+TrOCR(3.5GB) fit in 16GB
    r = EngineRouter(profile="high", max_engines=3, available_ram_gb=16.0)
    chosen, reasons = r.select(block_type="handwriting")
    assert chosen[0] == ENGINE_QWEN_HANDWRITTEN
    assert ENGINE_QARI in chosen


def test_diacritics_add_qari_for_arabic():
    r = EngineRouter(profile="balanced")
    chosen, _ = r.select(has_diacritics=True, language="ar")
    assert ENGINE_QARI in chosen


def test_table_block_adds_nougat():
    r = EngineRouter(profile="high")
    chosen, _ = r.select(block_type="table")
    assert ENGINE_NOUGAT in chosen


def test_arabic_adds_easyocr():
    r = EngineRouter(profile="balanced")
    chosen, _ = r.select(language="ar")
    assert ENGINE_EASYOCR in chosen


def test_low_quality_appends_tesseract():
    r = EngineRouter(profile="balanced")
    chosen, reasons = r.select(image_quality=0.4, language="ar")
    assert ENGINE_TESSERACT in chosen
    assert any("low image quality" in x for x in reasons)


def test_high_profile_adds_paddle_for_arabic():
    r = EngineRouter(profile="high")
    chosen, _ = r.select(language="ar")
    assert ENGINE_PADDLE in chosen


def test_latin_high_quality_prefers_trocr():
    r = EngineRouter(profile="high")
    chosen, _ = r.select(language="en", image_quality=0.9)
    assert chosen[0] == ENGINE_TROCR


def test_max_engines_respected_and_deduped():
    r = EngineRouter(profile="high", max_engines=2)
    chosen, reasons = r.select(block_type="handwriting", has_diacritics=True)
    assert len(chosen) <= 2
    assert len(chosen) == len(set(chosen))


def test_default_fallback_when_no_signal():
    # en language at low quality: no latin rule (quality<0.75), no arabic rule
    r = EngineRouter(profile="balanced")
    chosen, reasons = r.select(language="en", image_quality=0.5)
    assert chosen  # fallback still returns something


def test_ram_filter_constrains_heavy_engines():
    # TrOCR needs 3.5GB; 3GB budget must drop it
    r = EngineRouter(profile="high", available_ram_gb=3.0, max_engines=5)
    chosen, _ = r.select(language="en", image_quality=0.9)
    assert ENGINE_TROCR not in chosen


def test_ram_filter_empty_falls_back_to_tesseract():
    r = EngineRouter(profile="balanced", available_ram_gb=0.0)
    chosen, reasons = r.select(language="ar")
    assert chosen == [ENGINE_TESSERACT]
    assert reasons == ["RAM-constrained fallback"]


def test_registry_filters_allowed_engines():
    class Reg:
        _probed = True

        @staticmethod
        def available_engine_names():
            return [ENGINE_TESSERACT]

    r = EngineRouter(profile="high", registry=Reg())
    chosen, _ = r.select(language="ar")
    assert chosen == [ENGINE_TESSERACT]


def test_estimate_time_cpu_multiplier():
    r = EngineRouter(profile="balanced", use_gpu=False)
    # estimate_time rounds to one decimal: round(0.4*1.8, 1) == 0.7
    assert r.estimate_time([ENGINE_TESSERACT]) == pytest.approx(0.7)


def test_summary_roundtrip():
    r = EngineRouter(profile="low", use_gpu=True, max_engines=1, available_ram_gb=2.0)
    s = r.summary()
    assert s["profile"] == "low"
    assert s["use_gpu"] is True
    assert s["max_engines"] == 1
    assert s["available_ram_gb"] == 2.0
    assert ENGINE_TESSERACT in s["allowed_engines"]


def test_select_logs_decision_via_telemetry(monkeypatch):
    captured = {}

    def fake_log_decision(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(router_module, "log_decision", fake_log_decision)
    r = EngineRouter(profile="low")
    r.select()
    assert captured["decision"] == "engine_selection"
    assert captured["outcome"] == [ENGINE_TESSERACT]
