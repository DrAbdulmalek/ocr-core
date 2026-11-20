# tests/test_evidence_scope.py
"""اختبارات نطاق الدليل البصري (R16) — الملاحظة الهندسية على PR #57.

المبدأ المُختبَر: الدليل البصري يُثبت ما يُلحق به فقط.
- علامة واحدة في صفحة متعددة الإجابات لا تُصادق على الصفحة كلها.
- الدليل يُشتق من النص المُمرَّر نفسه، لا من إحصاءات متراكمة عبر النصوص.
- ما لا دليل عليه يبقى uncertain (المحافظة على حماية بيانات التدريب).
"""
import pytest

from ocr_core.rules.engine import OCRProcessor


@pytest.fixture()
def proc():
    return OCRProcessor()


# ---------- 1) علامة واحدة لا تُصادق على الصفحة كاملة ----------
def test_single_marker_does_not_certify_whole_page(proc):
    """صفحة بثلاث إجابات وعلامة ✓ بجوار الإجابة الثانية فقط:
    يجب أن يبقى bucket الصحيح محصورًا بسطر العلامة، والباقي uncertain."""
    page = (
        "Q1: نص الإجابة الاولى.\n"
        "Q2: نص الإجابة الثانيه. ✓.\n"
        "Q3: نص الإجابة الثالثة."
    )
    result = proc.process_text(page)
    # الدليل يخص سطر Q2 فقط
    assert "Q2" in result["correct"], (
        "سطر العلامة يجب أن يكون في bucket الصحيح"
    )
    # علامة واحدة لا تعمم على بقية الصفحة
    assert "Q1" not in result["correct"], (
        "علامة واحدة لا تصادق على إجابة بلا دليل"
    )
    assert "Q3" not in result["correct"], (
        "علامة واحدة لا تصادق على إجابة بلا دليل"
    )
    # ما بلا دليل يبقى غير محسوم
    assert "Q1" in result["uncertain"]
    assert "Q3" in result["uncertain"]
    # تصنيف المستند الكامل محافظ (خليط مستويات الدليل)
    assert result["classification"] == "uncertain"
    # لا شيء يُرسل إلى bucket الخطأ
    assert result["wrong"] == ""


# ---------- 2) علامتا صحة وخطأ في موضعين مختلفين ----------
def test_mixed_markers_route_to_separate_buckets(proc):
    """علامة ✓ في سطر و✗ في آخر: كل سطر يذهب لـ bucket دليله،
    والباقي uncertain — ولا يُطبع تصنيف كلي على الصفحة."""
    page = (
        "Q1: نص الإجابة الاولى. ✓.\n"
        "Q2: نص الإجابة الثانيه.\n"
        "Q3: نص الإجابة الثالثة. ✗.\n"
    )
    result = proc.process_text(page)
    assert "Q1" in result["correct"]
    assert "Q2" not in result["correct"] and "Q3" not in result["correct"]
    assert "Q3" in result["wrong"]
    assert "Q1" not in result["wrong"] and "Q2" not in result["wrong"]
    assert "Q2" in result["uncertain"]
    # دليل متضارب على مستوى المستند → غير محسوم (لا correct شامل)
    assert result["classification"] == "uncertain"


# ---------- 3) لا تسرب أدلة بين استدعاءات منفصلة ----------
def test_markers_do_not_leak_across_direct_calls(proc):
    """r16 يشتق الدليل من النص المُمرَّر نفسه: علامات نص سابق
    (متراكمة في stats) لا تصادق نصًا لاحقًا بلا علامات."""
    # نص سابق بعلامة صحة (استدعاء مباشر للقاعدتين، كما يفعلها مستهلك خارجي)
    proc.r14_detect_visual_markers("نص سابق صحيح. ✓")
    # نص لاحق بلا أي علامات — يجب أن يبقى uncertain مهما تراكم سابقًا
    sep = proc.r16_separate_correct_vs_wrong("نص عادي جديد بلا اي علامات.")
    assert sep["classification"] == "uncertain", (
        "أدلة نص سابق لا تصادق نصًا لاحقًا"
    )
    assert sep["correct"] == ""
    assert sep["wrong"] == ""


# ---------- 4) صفحة بلا علامات تبقى uncertain بالكامل (حارس انحدار) ----------
def test_unmarked_multiline_page_stays_uncertain(proc):
    result = proc.process_text("سطر اول عادي تماما.\nسطر ثان عادي ايضا.")
    assert result["classification"] == "uncertain"
    assert result["correct"] == ""
    assert result["wrong"] == ""
    assert "سطر اول" in result["uncertain"]


# ---------- 5) X الغامضة لا تؤكد الخطأ في سياق متعدد الأسطر ----------
def test_ambiguous_x_not_negative_in_multiline_context(proc):
    result = proc.process_text(
        "السطر الاول يذكر X-ray صراحة.\nالسطر الثاني عادي."
    )
    assert result["classification"] == "uncertain"
    assert result["wrong"] == ""
    # X/x مستثناة عمدًا من قائمة علامات R14
    assert "X" not in result["metadata"]["markers"]


# ---------- 6) توثيق بوابة أهلية التدريب للأمثلة السلبية المؤكدة ----------
def test_confirmed_negative_stays_candidate_until_policy_decision(proc):
    """مثال سلبي مؤكد بشريًا بعلامة ✗ ومرجع مصدر: status=verified
    ويُحفظ كـ negative_example، لكن training_eligible تبقى False
    وفق البوابة الحالية (الإيجابي المؤكد فقط يُرقى) — توثيق صريح
    لقرار السياسة في تقرير التدقيق."""
    rec = proc.build_training_record(
        source="المصدر الأصلي",
        correct="",
        incorrect="ترجمة مرشحة خاطئة",
        evidence={
            "confidence": "high",
            "visual_marker": "✗",
            "label": "negative_example",
            "source_reference": "document.pdf#page=3",
            "evidence_confirmed": True,
        },
    )
    assert rec["status"] == "verified"
    assert rec["negative_example"] == "ترجمة مرشحة خاطئة"
    assert rec["positive_example"] is None
    # البوابة الحالية: أهلية التدريب للإيجابي المؤكد فقط
    assert rec["training_eligible"] is False
