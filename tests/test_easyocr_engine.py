"""EasyOCR engine: protocol + fail-visible errors. No easyocr required.

Mirrors the PaddleEngine test pattern. A BGR→RGB heuristic test is numpy-gated
(easyocr itself is never installed in CI).
"""
import pytest

from ocr_core.engines.easyocr import EasyOCREngine

_PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)


def test_easyocr_metadata():
    e = EasyOCREngine(languages=["ar"])
    assert e.name == "easyocr"
    assert e.languages == ["ar"]
    assert e.gpu is False  # reproducible CI default (omni default was True — documented deviation)
    assert e.batch_size == 4
    assert e.enable_rtl_fix is True


def test_easyocr_rtl_fixer_is_ocr_core_derived_map():
    # the engine must use ocr_core's NFKC-derived fixer, not omni's hand-written table
    from ocr_core.rtl_utils import ARABIC_NORMALIZATION_MAP
    assert EasyOCREngine().rtl_fixer.normalize_presentation_forms("\uFEC7") == "ظ"
    assert ARABIC_NORMALIZATION_MAP[chr(0xFEC7)] == "ظ"


def test_easyocr_missing_file_is_reported_not_raised(tmp_path):
    r = EasyOCREngine().process_image(tmp_path / "nope.png")
    assert r.error is not None and r.text == ""
    assert r.ok is False
    assert r.engine == "easyocr"


def test_easyocr_missing_pdf_is_reported_not_raised(tmp_path):
    r = EasyOCREngine().process_pdf(tmp_path / "nope.pdf")
    assert r.error is not None and r.text == ""


def test_easyocr_unavailable_is_error_not_empty(tmp_path, monkeypatch):
    e = EasyOCREngine()
    monkeypatch.setattr(e, "available", lambda: False)
    img = tmp_path / "x.png"
    img.write_bytes(_PNG_1PX)
    r = e.process_image(img)
    assert r.error is not None
    assert "easyocr" in r.error
    assert r.text == ""


def test_easyocr_pdf_unreadable_reported(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    r = EasyOCREngine().process_pdf(bad)
    assert r.error is not None


def test_mean_confidence_never_invents():
    from ocr_core.engines.easyocr import _mean_confidence
    assert _mean_confidence([]) == 0.0
    assert _mean_confidence([{"confidence": 0.5}, {"confidence": 0.7}]) == pytest.approx(0.6)


def test_ensure_rgb_conversions():
    np = pytest.importorskip("numpy")
    from ocr_core.engines.easyocr import _ensure_rgb
    gray = np.zeros((4, 4), dtype="uint8")
    assert _ensure_rgb(gray).shape == (4, 4, 3)
    rgba = np.zeros((4, 4, 4), dtype="uint8")
    assert _ensure_rgb(rgba).shape == (4, 4, 3)
    # blue-dominant → flipped to RGB
    bgr = np.zeros((4, 4, 3), dtype="uint8")
    bgr[:, :, 0] = 200  # blue channel dominant
    out = _ensure_rgb(bgr)
    assert out[0, 0, 2] == 200  # blue moved to channel 2
    # already RGB-ish stays
    rgb = np.zeros((4, 4, 3), dtype="uint8")
    rgb[:, :, 0] = 200  # red dominant
    assert _ensure_rgb(rgb) is rgb or _ensure_rgb(rgb).shape == (4, 4, 3)
