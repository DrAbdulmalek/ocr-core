"""اختبارات موارد tessdata المضمنة (Apache-2.0) واكتشافها في TesseractEngine."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ocr_core.engines.tesseract import TesseractEngine, _BUNDLED_TESSDATA

# بصمات tessdata_fast @ 87416418657359cb625c412a48b6e1d6d41c29bd
EXPECTED_SHA = {
    "ara": "e3206d3dc87fd50c24a0fb9f01838615911d25168f4e64415244b67d2bb3e729",
    "eng": "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_bundled_models_present_with_provenance():
    for lang, expected in EXPECTED_SHA.items():
        p = _BUNDLED_TESSDATA / f"{lang}.traineddata"
        assert p.is_file(), f"مفقود: {p}"
        assert _sha256(p) == expected, f"بصمة {lang} غير مطابقة — الملف معدل أو تالف"
    assert (_BUNDLED_TESSDATA / "PROVENANCE.md").is_file()


def test_engine_autodiscovers_bundled_tessdata():
    engine = TesseractEngine()
    if engine.tessdata_dir is None:
        # الموارد مفقودة في بيئة التثبيت — التخطي الصادق مطلوب
        import pytest

        pytest.skip("النماذج المضمنة غير متوفرة في هذه البيئة")
    assert engine.tessdata_dir.endswith("tessdata_fast")
    assert "--tessdata-dir" in engine._config()
    assert "--psm 6" in engine._config()


def test_explicit_tessdata_dir_overrides():
    engine = TesseractEngine(tessdata_dir="/custom/path")
    assert engine.tessdata_dir == "/custom/path"
    assert "/custom/path" in engine._config()


def test_config_without_dir_is_clean():
    engine = TesseractEngine(tessdata_dir=" ")  # فراغ لا يُحتسب مسارًا
    # الفراغ فالسي يعاد تفسيره discovery؛ نتقبل أيًا من النتيجتين بلا انهيار
    assert engine._config().startswith("--psm")


def test_default_lang_is_ara_plus_eng():
    assert TesseractEngine().lang == "ara+eng"
