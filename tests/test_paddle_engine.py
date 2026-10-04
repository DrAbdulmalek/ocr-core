"""Paddle engine: protocol + fail-visible errors. No paddleocr required."""
from ocr_core.engines.paddle import PaddleEngine


def test_paddle_metadata():
    e = PaddleEngine(lang="ar")
    assert e.name == "paddle"
    assert e.lang == "ar"
    assert e.use_gpu is False


def test_paddle_missing_file_is_reported_not_raised(tmp_path):
    r = PaddleEngine().process_image(tmp_path / "nope.png")
    assert r.error is not None and r.text == ""
    assert r.ok is False
    assert r.engine == "paddle"


def test_paddle_missing_pdf_is_reported_not_raised(tmp_path):
    r = PaddleEngine().process_pdf(tmp_path / "nope.pdf")
    assert r.error is not None and r.text == ""


def test_paddle_unavailable_is_error_not_empty(tmp_path, monkeypatch):
    e = PaddleEngine()
    monkeypatch.setattr(e, "available", lambda: False)
    img = tmp_path / "x.png"
    img.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
        )
    )
    r = e.process_image(img)
    assert r.error is not None
    assert "paddleocr" in r.error
    assert r.text == ""
