import pytest
from ocr_core.commands.core import Executor, ExecutionContext
from ocr_core.commands.builtins import io, preprocess, ocr, postprocess, export, benchmark  # noqa


def test_commands_registered():
    from ocr_core.commands.core import registry
    ids = {c.id for c in registry.list()}
    # تحقق من وجود الأوامر الجوهرية
    assert "io.load_image" in ids
    assert "preprocess.deskew" in ids
    assert "ocr.extract" in ids
    assert "postprocess.correct_ar" in ids
    assert "export.markdown" in ids
    assert "benchmark.cer" in ids


def test_command_count():
    from ocr_core.commands.core import registry
    assert len(registry.list()) >= 28


def test_categories():
    from ocr_core.commands.core import registry
    cats = registry.categories()
    for expected in ["io", "preprocess", "ocr", "postprocess", "export", "benchmark"]:
        assert expected in cats


def test_cer_calculation():
    ex = Executor()
    ctx = ExecutionContext()
    ctx.set("corrected_text", "مرحبا")
    r = ex.execute("benchmark.cer",
                   {"prediction_key": "corrected_text",
                    "reference": "مرحبا"},
                   ctx)
    assert r["ok"] is True
    assert r["result"]["cer"] == 0.0


def test_wer_calculation():
    ex = Executor()
    ctx = ExecutionContext()
    ctx.set("corrected_text", "hello world")
    r = ex.execute("benchmark.wer",
                   {"prediction_key": "corrected_text",
                    "reference": "hello world"},
                   ctx)
    assert r["ok"] is True
    assert r["result"]["wer"] == 0.0


def test_guardrails_reverts_on_digit_change():
    ex = Executor()
    ctx = ExecutionContext()
    ctx.set("ocr_result", {"text": "جرعة 500 mg"})
    ctx.set("corrected_text", "جرعة ٥٠٠ mg")  # الأرقام تغيرت
    r = ex.execute("postprocess.guardrails", {}, ctx)
    assert r["result"]["digits_preserved"] is False
    assert r["result"]["reverted"] is True
    assert ctx.get("corrected_text") == "جرعة 500 mg"
