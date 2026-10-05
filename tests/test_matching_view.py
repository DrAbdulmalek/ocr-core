"""Tests for ocr_core.postprocess.matching_view — non-destructive matching view.

Covers (clean-room, original algorithms):
- non-destructive contract: original_text is byte-for-byte preserved
- rule application with per-rule audit counts (aligned with normalization.py)
- medical dose protection (mask → normalize → restore verbatim)
- opt-in taa-marbuta unification (metric-bias warning honored)
- idempotency and whitespace collapse accounting

Run with:
    pytest tests/test_matching_view.py -v
"""

from ocr_core.postprocess.matching_view import (
    to_matching_view,
)

# ── العقد غير التدميري ───────────────────────────────────────────────────────


def test_empty_string_is_noop():
    result = to_matching_view("")
    assert result.original_text == ""
    assert result.matching_text == ""
    assert result.changes == ()


def test_original_text_preserved_verbatim():
    raw = "أَخَذَ الــمريضُ ٥٠٠ ملغ صباحاً"
    result = to_matching_view(raw)
    assert result.original_text == raw  # byte-for-byte، بلا أي تلميع
    assert result.matching_text != raw or result.changes == ()


def test_latin_only_text_yields_no_changes():
    result = to_matching_view("Patient 250 mg twice daily")
    assert result.matching_text == result.original_text
    assert result.changes == ()


# ── القواعد وحساب التغييرات ─────────────────────────────────────────────────


def test_alef_unification_with_count():
    result = to_matching_view("أإآ")
    assert result.matching_text == "ااا"
    assert result.change_counts()["unify_alef"] == 3


def test_alef_maqsura_unified_to_ya():
    result = to_matching_view("على")
    assert result.matching_text == "علي"  # same documented behavior as normalization.py
    assert result.change_counts()["unify_alef_maqsura"] == 1


def test_diacritics_removed():
    result = to_matching_view("صَيدَلِيَّةٌ")
    assert result.matching_text == "صيدلية"  # ة تبقى: قاعدة التاء اختيارية
    assert "remove_diacritics" in result.change_counts()


def test_tatweel_removed():
    result = to_matching_view("الــسلام")
    assert result.matching_text == "السلام"
    assert result.change_counts()["remove_tatweel"] == 1


def test_arabic_indic_digits_converted():
    result = to_matching_view("ضغط ١٢٠")
    assert result.matching_text == "ضغط 120"
    assert result.change_counts()["unify_arabic_digits"] == 3


def test_whitespace_collapse_is_recorded():
    result = to_matching_view("سطر   مبعثر\tجدا")
    assert result.matching_text == "سطر مبعثر جدا"
    assert result.change_counts()["collapse_whitespace"] == 1


# ── التاء المربوطة: اختيارية مع تحذير موثق ──────────────────────────────────


def test_taa_marbuta_disabled_by_default():
    result = to_matching_view("صيدلية")
    assert result.matching_text == "صيدلية"
    assert "unify_taa_marbuta" not in result.change_counts()


def test_taa_marbuta_opt_in_and_note():
    result = to_matching_view("صيدلية", taa_marbuta=True)
    assert result.matching_text == "صيدليه"
    change = next(c for c in result.changes if c.rule == "unify_taa_marbuta")
    assert change.note  # التحذير (انحياز المقاييس) مرفق بالسجل


# ── حماية المقادير الطبية (فعلية وليست شكلية) ───────────────────────────────


def test_arabic_dose_protected_from_digit_rule():
    result = to_matching_view("خذ ٥٠٠ ملغ مرتين")
    assert "٥٠٠ ملغ" in result.matching_text  # digits NOT converted inside dose
    assert "unify_arabic_digits" not in result.change_counts()


def test_protection_disabled_applies_rules_inside_dose():
    result = to_matching_view("خذ ٥٠٠ ملغ مرتين", protect_medical=False)
    assert "500 ملغ" in result.matching_text
    assert result.change_counts()["unify_arabic_digits"] == 3


def test_celsius_dose_protected_verbatim():
    result = to_matching_view("الحرارة ٣٧.٥ °C")
    assert "٣٧.٥ °C" in result.matching_text


def test_latin_dose_preserved_byte_exact():
    result = to_matching_view("Amoxicillin 500 mg", taa_marbuta=True)
    assert "500 mg" in result.matching_text


def test_multiple_doses_restored_in_order():
    result = to_matching_view("٥٠ ملغ ثم ٢٥٠ ملغ")
    assert "٥٠ ملغ ثم ٢٥٠ ملغ" in result.matching_text


def test_fractional_arabic_dose_protected():
    result = to_matching_view("٠.٥ مل مرة واحدة")
    assert "٠.٥ مل" in result.matching_text


# ── الاتساق والانعدامية ─────────────────────────────────────────────────────


def test_idempotent_second_pass_default_rules():
    first = to_matching_view("أَخذَ ٥٠٠ ملغ صباحاً")
    second = to_matching_view(first.matching_text)
    assert second.change_counts() == {}


def test_idempotent_second_pass_with_taa_marbuta():
    first = to_matching_view("صيدلية عامة", taa_marbuta=True)
    second = to_matching_view(first.matching_text, taa_marbuta=True)
    assert "unify_taa_marbuta" not in second.change_counts()


def test_mixed_arabic_english_medical_line():
    raw = "التشخيص: Infection — 500 mg كل 8 ساعات"
    result = to_matching_view(raw)
    assert result.original_text == raw
    assert "500 mg" in result.matching_text
    assert "التشخيص" in result.matching_text  # لا يُمس: بلا قواعد تنطبق عليه


def test_change_counts_matches_changes_tuple():
    result = to_matching_view("أَ ٥٠ ملغ ب")
    assert result.change_counts() == {c.rule: c.count for c in result.changes}
