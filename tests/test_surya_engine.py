"""اختبارات محرك Surya — تُتخطى بأناقة عند غياب الحزمة، ولا تُحمّل نماذج أبدًا."""
from __future__ import annotations

import importlib.util

import pytest


def _surya_installed() -> bool:
    return importlib.util.find_spec("surya") is not None


def test_surya_not_installed_graceful():
    """عند غياب surya: رسالة واضحة عبر OCRResult.error — بلا استثناء."""
    from ocr_core.engines.surya_engine import SuryaEngine

    if _surya_installed():
        pytest.skip("surya مثبت — لا يمكن محاكاة الغياب هنا")
    engine = SuryaEngine()
    assert engine.is_available() is False
    r = engine.process_image("whatever.png")
    assert r.error is not None
    assert "surya" in r.error.lower()


def test_is_available_static_bool():
    """is_available() يعيد bool ولا يرفع استثناء أبدًا."""
    from ocr_core.engines.surya_engine import SuryaEngine

    assert isinstance(SuryaEngine.is_available(), bool)


def test_missing_file_error():
    """ملف غير موجود → error نصي، لا استثناء (سياسة base.py)."""
    from ocr_core.engines.surya_engine import SuryaEngine

    engine = SuryaEngine()
    r = engine.process_image("/nonexistent/path/page.png")
    assert r.error is not None
    assert "file not found" in r.error


def test_engine_name_and_meta():
    from ocr_core.engines.surya_engine import SuryaEngine

    engine = SuryaEngine(langs_hint=["ar", "en"])
    assert engine.name == "surya"
    assert engine.langs_hint == ["ar", "en"]


def test_html_to_text_basic():
    """تحويل HTML→نص يعمل بلا اعتماديات (أصلي، stdlib فقط)."""
    from ocr_core.engines.surya_engine import _html_to_text

    assert _html_to_text("<p>مرحبا</p><p>world</p>").count("\n") >= 1
    assert "مرحبا" in _html_to_text("<p>مرحبا &amp; سلام</p>")
    assert _html_to_text("") == ""


@pytest.mark.skipif(not _surya_installed(), reason="surya-ocr غير مثبت")
def test_surya_import_and_version_guard():
    """مع التثبيت: الاستيراد يعمل وتحذير الإصدار يعمل دون تحميل نماذج."""
    from ocr_core.engines import surya_engine
    from ocr_core.engines.surya_engine import SuryaEngine

    assert SuryaEngine.name == "surya"
    # التحذير إما None (0.22+) أو نص تحذيري — لا استثناء
    warn = SuryaEngine._version_warning()
    assert warn is None or isinstance(warn, str)
    assert surya_engine._SURYA_TARGET == (0, 22)


@pytest.mark.skipif(not _surya_installed(), reason="surya-ocr غير مثبت")
def test_process_image_reports_errors_not_raises(tmp_path):
    """صورة فاسدة → OCRResult.error (لا استثناء) حتى مع surya مثبتًا."""
    from ocr_core.engines.surya_engine import SuryaEngine

    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not-an-image")
    engine = SuryaEngine()
    r = engine.process_image(str(bad))
    # إما error من PIL أو نتيجة — المهم: لا استثناء خارج OCRResult
    assert r.error is None or isinstance(r.error, str)
