"""ArabicRTLFixer: truth-anchored Unicode gates + reversal-fix behavior.

The map gate is the lesson of omni#162: every map entry must equal NFKC of its
key — derived at import time here, but asserted anyway so a future edit that
replaces the derivation with a hand-written table cannot drift silently.
"""
import unicodedata

from ocr_core.rtl_utils import (
    ARABIC_NORMALIZATION_MAP,
    ArabicRTLFixer,
    RTLFixStats,
)


def test_map_covers_all_standard_letter_forms():
    # 124 presentation forms in U+FE81..U+FEFC whose NFKC is a standard letter
    expected = 0
    for cp in range(0xFE81, 0xFEFD):
        nfkc = unicodedata.normalize("NFKC", chr(cp))
        if all(0x0621 <= ord(c) <= 0x064A for c in nfkc):
            expected += 1
            assert chr(cp) in ARABIC_NORMALIZATION_MAP, f"missing U+{cp:04X}"
    assert len(ARABIC_NORMALIZATION_MAP) == expected


def test_every_map_entry_equals_nfkc():
    for key, mapped in ARABIC_NORMALIZATION_MAP.items():
        assert mapped == unicodedata.normalize("NFKC", key), (
            f"U+{ord(key):04X} maps to {mapped!r}, NFKC says "
            f"{unicodedata.normalize('NFKC', key)!r}"
        )


def test_ranges_that_were_drifted_in_omni_162():
    # Regression for the omni#162 off-by-2 bug classes (never reproduce here):
    assert ARABIC_NORMALIZATION_MAP[chr(0xFEC3)] == "ط"  # Tah initial (was ظ)
    assert ARABIC_NORMALIZATION_MAP[chr(0xFEC7)] == "ظ"  # Zah initial (was ع)
    assert ARABIC_NORMALIZATION_MAP[chr(0xFEC8)] == "ظ"  # Zah medial (was ع)
    assert ARABIC_NORMALIZATION_MAP[chr(0xFECB)] == "ع"  # Ain initial (was غ)
    assert ARABIC_NORMALIZATION_MAP[chr(0xFECF)] == "غ"  # Ghain initial (was ف)
    assert ARABIC_NORMALIZATION_MAP[chr(0xFED3)] == "ف"  # Feh initial (was ق)
    assert ARABIC_NORMALIZATION_MAP[chr(0xFEDD)] == "ل"  # Lam isolated (was ك)
    assert ARABIC_NORMALIZATION_MAP[chr(0xFEDE)] == "ل"  # Lam final (was ك)
    assert ARABIC_NORMALIZATION_MAP[chr(0xFED7)] == "ق"  # Qaf initial (missing in omni)
    assert ARABIC_NORMALIZATION_MAP[chr(0xFED9)] == "ك"  # Kaf isolated (missing in omni)


def test_lam_alef_ligatures_two_char_nfkc():
    # omni's hand-written table had these four wrong (order drift); truth = NFKC:
    assert ARABIC_NORMALIZATION_MAP[chr(0xFEF5)] == "لآ"  # LAM + ALEF WITH MADDA ABOVE
    assert ARABIC_NORMALIZATION_MAP[chr(0xFEF7)] == "لأ"  # LAM + ALEF WITH HAMZA ABOVE
    assert ARABIC_NORMALIZATION_MAP[chr(0xFEF9)] == "لإ"  # LAM + ALEF WITH HAMZA BELOW
    assert ARABIC_NORMALIZATION_MAP[chr(0xFEFB)] == "لا"  # LAM + ALEF


def test_normalize_presentation_forms_full_line():
    fixer = ArabicRTLFixer()
    # presentation forms of تشخيص (isolated forms FE95 FE97 EE... simplified: use final forms)
    line = "ﺗﺸﺨﻴﺺ"  # Tah/Sheen/Khah/Yeh/Sad final forms
    out = fixer.normalize_presentation_forms(line)
    assert out == "تشخيص"
    assert not any(0xFB50 <= ord(c) <= 0xFEFF for c in out)


def test_contains_arabic():
    assert ArabicRTLFixer.contains_arabic("مرحبا")
    assert ArabicRTLFixer.contains_arabic("mixed ﻋربي")
    assert not ArabicRTLFixer.contains_arabic("hello world 123")


def test_reversal_ratio_flags_reversed_medical_text():
    fixer = ArabicRTLFixer()
    correct = "المريض يشكو من ارتفاع ضغط الدم وتسارع ضربات القلب"
    reversed_text = " " .join(t[::-1] for t in correct.split())
    assert fixer.reversal_ratio(correct) <= 0.2
    assert fixer.reversal_ratio(reversed_text) >= fixer.reversal_threshold


def test_fix_text_reverses_reversed_arabic_keeps_latin():
    fixer = ArabicRTLFixer()
    # one reversed Arabic token + a Latin token that must stay intact
    reversed_token = "صيخشت"  # "تشخيص" reversed
    out = fixer.fix_text(f"diagnosis: {reversed_token}", force=True)
    assert "تشخيص" in out
    assert "diagnosis:" in out


def test_fix_text_no_op_for_correct_arabic():
    fixer = ArabicRTLFixer()
    correct = "المريض يشكو من ارتفاع ضغط الدم"
    assert fixer.fix_text(correct) == correct


def test_analyze_and_fix_stats():
    fixer = ArabicRTLFixer()
    fixed, stats = fixer.analyze_and_fix("صيخشت", force=True)
    assert isinstance(stats, RTLFixStats)
    assert stats.changed is True
    assert fixed == "تشخيص"


def test_empty_input():
    fixer = ArabicRTLFixer()
    assert fixer.fix_text("") == ""
    assert fixer.normalize_presentation_forms("") == ""
