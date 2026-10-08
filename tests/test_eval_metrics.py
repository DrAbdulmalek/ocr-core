"""اختبارات وحدة التقييم القانونية (F-13/F-16): metrics + normalize_v1 + harness.

تغطي: دلالات skip_internal_normalize (منع التطبيع المزدوج)، توافق السلوك
الافتراضي مع النسخة المرجعية، سياسة normalize_v1 المعلنة، و smoke-test
للـ harness عبر subprocess.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_core.eval import (
    NORMALIZE_V1_VERSION,
    calculate_cer,
    calculate_wer,
    normalize_v1,
)
from ocr_core.eval.metrics import _normalize_arabic


# --------------------------------------------------------------- CER أساسي
def test_cer_identical_is_zero():
    cer, edits, total = calculate_cer("نص مطابق تماما", "نص مطابق تماما")
    assert (cer, edits) == (0.0, 0)
    assert total == len("نص مطابق تماما")


def test_cer_empty_reference_edge():
    # مرجع فارغ + فرضية غير فارغة → 1.0 (كل شيء خطأ) — نفس دلالات النسخة المرجعية
    assert calculate_cer("", "abc") == (1.0, 3, 0)
    # كلاهما فارغ → 0.0
    assert calculate_cer("", "") == (0.0, 0, 0)


def test_cer_all_wrong_is_one():
    cer, edits, total = calculate_cer("abcdef", "xyz")
    assert cer == 1.0
    assert edits == 6 and total == 6


# --------------------------------------------- F-13/F-16: skip_internal_normalize
def test_default_normalizes_internally_hamza_forms_match():
    # إ/أ/ا تتطابق بعد التطبيع الداخلي → صفر خطأ افتراضياً
    cer, edits, _ = calculate_cer("إسلام", "اسلام")
    assert (cer, edits) == (0.0, 0)


def test_skip_true_disables_internal_normalization():
    # مع skip=True: التشكيل والهمزات تُحسب أخطاءً — يثبت أن المفتاح يغيّر السلوك
    cer_skipped, edits_skipped, _ = calculate_cer("إسلام", "اسلام", skip_internal_normalize=True)
    assert cer_skipped > 0.0 and edits_skipped > 0
    cer_default, _, _ = calculate_cer("إسلام", "اسلام")
    assert cer_default == 0.0


def test_skip_true_with_pre_normalized_inputs_is_zero():
    # المسار الإلزامي: طبّع الطرفين عبر normalize_v1 ثم استدعِ بـ True.
    # الفرق هنا تشكيلي فقط (normalize_v1 يزيله افتراضياً؛ طي الهمزة اختياري).
    ref_n, _ = normalize_v1("مُحَمَّد")
    hyp_n, _ = normalize_v1("محمد")
    cer, edits, _ = calculate_cer(ref_n, hyp_n, skip_internal_normalize=True)
    assert (cer, edits) == (0.0, 0)


def test_double_normalization_is_prevented_harness_path():
    # F-13: harness يطبّع مرة واحدة؛ نتيجته تختلف عن تطبيع مزدوج إذا اختلفت
    # السياسات — هنا نثبت أن المسار الموحد يعطي القيمة المتوقعة حرفياً.
    ref = "مُحَمَّد ١٢٣"
    hyp = "محمد 123"
    ref_n, _ = normalize_v1(ref)
    hyp_n, _ = normalize_v1(hyp)
    cer, _, _ = calculate_cer(ref_n, hyp_n, skip_internal_normalize=True)
    assert cer == 0.0  # التشكيل زال والأرقام وُحّدت عبر normalize_v1


# ------------------------------------------------------------------ WER
def test_wer_word_level():
    wer, edits, total = calculate_wer("الطبيب هنا", "الطبيب هناك")
    assert wer == pytest.approx(0.5)
    assert edits == 1 and total == 2


def test_wer_empty_reference_edge():
    assert calculate_wer("", "كلمة") == (1.0, 1, 0)
    assert calculate_wer("", "") == (0.0, 0, 0)


# ---------------------------------------------------------- _normalize_arabic توافق
def test_internal_normalize_matches_reference_semantics():
    assert _normalize_arabic("أَحْمَد") == "احمد"
    assert _normalize_arabic("مدرسة   جديدة") == "مدرسة جديدة"
    assert _normalize_arabic(None) == ""


# ------------------------------------------------------------- normalize_v1
def test_normalize_v1_policy_and_version():
    out, policy = normalize_v1("مُحَمَّد")
    assert policy["version"] == NORMALIZE_V1_VERSION == "normalize_v1/1.0.0"
    assert policy["changed"] is True
    assert policy["fold_hamza"] is False
    assert "remove_diacritics_U+064B-U+065F_U+0670" in policy["steps"]
    assert out == "محمد"


def test_normalize_v1_digits_tatweel_zero_width():
    out, _ = normalize_v1("١٢٣ ـــ كتاب\u200f")
    assert "1" in out and "3" in out
    assert "\u0640" not in out and "\u200f" not in out


def test_normalize_v1_none_and_fold_hamza_optional():
    out, policy = normalize_v1(None)
    assert out == "" and policy["changed"] is False
    out2, policy2 = normalize_v1("أؤمن", fold_hamza=True)
    assert policy2["fold_hamza"] is True and "fold_hamza" in policy2["steps"]
    assert out2 == "اومن"


# ----------------------------------------------------------- harness smoke
def test_golden_harness_subprocess_smoke(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    rows = [
        {"id": "a", "reference": "مُحَمَّد", "hypothesis": "محمد"},
        {"id": "b", "reference": "كتاب ١٢٣", "hypothesis": "كتاب 123"},
    ]
    pairs.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "golden_harness.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(pairs), "--json"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["aggregate"]["pairs"] == 2
    assert payload["aggregate"]["skip_internal_normalize"] is True
    assert payload["aggregate"]["normalize_version"] == "normalize_v1/1.0.0"
    by_id = {r["id"]: r for r in payload["items"]}
    assert by_id["a"]["cer"] == 0.0
    assert by_id["b"]["cer"] == 0.0
