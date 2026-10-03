"""Stage 5 tests: engine base protocol + optional Tesseract engine."""
import shutil

import pytest

from ocr_core.engines.base import OCRResult, OCREngine


def test_ocr_result_defaults():
    r = OCRResult()
    assert r.text == "" and r.engine == "" and r.confidence == 0.0
    assert r.error is None and r.ok is True and r.pages == 1


def test_ocr_result_error_flags_not_ok():
    r = OCRResult(error="boom")
    assert r.ok is False


def test_base_engine_not_available_and_raises():
    e = OCREngine()
    assert e.available() is False
    with pytest.raises(NotImplementedError):
        e.process_image("x.png")
    with pytest.raises(NotImplementedError):
        e.process_pdf("x.pdf")


def test_tesseract_metadata():
    from ocr_core.engines.tesseract import TesseractEngine

    e = TesseractEngine(lang="eng", psm="7")
    assert e.name == "tesseract" and e.lang == "eng" and e.psm == "7"


def test_tesseract_missing_file_is_reported_not_raised(tmp_path):
    from ocr_core.engines.tesseract import TesseractEngine

    r = TesseractEngine().process_image(tmp_path / "nope.png")
    assert r.error is not None and r.text == ""


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary not installed")
def test_tesseract_end_to_end_english(tmp_path):
    pytest.importorskip("pytesseract")
    from PIL import Image, ImageDraw, ImageFont

    from ocr_core.engines.tesseract import TesseractEngine

    img = Image.new("RGB", (420, 120), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    d.text((20, 40), "HELLO 123", fill="black", font=font)
    p = tmp_path / "sample.png"
    img.save(p)

    r = TesseractEngine(lang="eng", psm="7").process_image(p)
    assert r.error is None, r.error
    assert "HELLO" in r.text
    assert r.confidence == 0.0  # never invented
    assert r.processing_time > 0