"""Tests for ocr_core.postprocess medical modules (stage-4 port from omni).

- normalization unit checks
- field_extractor extraction + fingerprint stability
- WeightedMedicalDeduplicator scenarios adapted from omni
  scripts/test_phase4_field_dedup.py (print-checks converted to asserts)
- DeduplicationPipeline smoke
"""
from pathlib import Path

import ocr_core.postprocess.normalization as norm_mod
from ocr_core.postprocess.deduplication import (
    WeightedMedicalDeduplicator,
    field_aware_similarity,
)
from ocr_core.postprocess.deduplication_pipeline import DeduplicationPipeline
from ocr_core.postprocess.field_extractor import ArabicMedicalFieldExtractor
from ocr_core.postprocess.normalization import (
    arabic_normalize,
    arabic_strong_normalize,
    normalize_batch,
)

# ── normalization ────────────────────────────────────────────────────────────


def test_arabic_normalize_unifies_digits():
    # documented behavior: ى→ي also applies (على → علي)
    assert arabic_normalize("ضغط الدم ١٢٠ على ٨٠") == "ضغط الدم 120 علي 80"


def test_arabic_normalize_unifies_alef_and_strips_diacritics():
    assert arabic_normalize("أَحْمَد") == "احمد"


def test_arabic_normalize_taa_marbuta_alef_maqsura_tatweel():
    assert arabic_normalize("مدرسة على") == "مدرسه علي"
    assert arabic_normalize("ـــاحمد") == "احمد"


def test_arabic_normalize_empty():
    assert arabic_normalize("") == ""


def test_strong_normalize_applies_medical_dict():
    # dict maps the alef-unified key to the hamza-corrected term
    assert arabic_strong_normalize("اموكسيسيلين") == "أموكسيسيلين"


def test_strong_normalize_without_dict_leaves_word():
    assert arabic_strong_normalize("اموكسيسيلين", use_medical_dict=False) == "اموكسيسيلين"


def test_normalize_batch():
    assert normalize_batch(["١", "أ"]) == ["1", "ا"]


def test_medical_terms_json_is_packaged():
    p = Path(norm_mod.__file__).parent / "data" / "medical_terms.json"
    assert p.exists(), "medical_terms.json must ship as package data"


# ── field_extractor ──────────────────────────────────────────────────────────

REPORT_A = """رقم الملف: MRN-2026-1007
اسم المريض: أحمد محمد العلي
تاريخ الميلاد: 15/03/1965
التاريخ: 2026-07-12
التشخيص: ارتفاع ضغط الدم الأساسي
الأدوية: أملوديبين 5mg، أسبرين 81mg
الطبيب المعالج: د. خالد الرشيدي
"""


def test_extract_fields_finds_core_fields():
    fields = ArabicMedicalFieldExtractor().extract_fields(REPORT_A)
    assert fields.patient_name
    assert fields.patient_id
    assert fields.diagnosis
    assert fields.medications


def test_patient_fingerprint_is_stable_and_content_bound():
    ex = ArabicMedicalFieldExtractor()
    f1 = ex.extract_fields(REPORT_A)
    f2 = ex.extract_fields(REPORT_A)
    assert f1.unique_patient_fingerprint()
    assert f1.unique_patient_fingerprint() == f2.unique_patient_fingerprint()
    # different patient -> different fingerprint
    other = REPORT_A.replace("أحمد محمد العلي", "فاطمة سعيد الحربي")
    f3 = ex.extract_fields(other)
    assert f3.unique_patient_fingerprint() != f1.unique_patient_fingerprint()


# ── deduplication (phase-4 scenarios) ────────────────────────────────────────

SAME_TEMPLATE_A = """
مستشفى الملك فهد التخصصي
قسم الباطنية
رقم الملف: MRN-2026-1007
اسم المريض: أحمد محمد العلي
تاريخ الميلاد: 15/03/1965
التاريخ: 2026-07-12
التشخيص: ارتفاع ضغط الدم الأساسي
الأدوية: أملوديبين 5mg، أسبرين 81mg
الطبيب المعالج: د. خالد الرشيدي
"""

SAME_TEMPLATE_B = """
مستشفى الملك فهد التخصصي
قسم الباطنية
رقم الملف: MRN-2026-2044
اسم المريض: فاطمة سعيد الحربي
تاريخ الميلاد: 22/08/1978
التاريخ: 2026-07-12
التشخيص: ارتفاع ضغط الدم الأساسي
الأدوية: أملوديبين 5mg، لوسارتان 50mg
الطبيب المعالج: د. خالد الرشيدي
"""

DOC_ORIGINAL = """
اسم المريض: أحمد محمد العلي
رقم المريض: MRN-2026-1007
التاريخ: 2026-07-12
التشخيص: ارتفاع ضغط الدم الأساسي
الأدوية: أملوديبين 5mg، أسبرين 81mg
"""

DOC_OCR_VARIANT = """
اسم المريض: احمد محمد العلي
رقم المريض: MRN-2026-1007
التاريخ: 2026-07-12
التشخيص: ارتفاع ضغط الدم الاساسي
الادوية: املوديبين 5mg، اسبرين 81mg
"""

DOC_DIFFERENT = """
اسم المريض: سارة عبدالله القحطاني
رقم المريض: MRN-2026-8899
التاريخ: 2026-06-01
التشخيص: التهاب الجيوب الأنفية المزمن
الأدوية: أموكسيسيلين 500mg، سودوافيدرين
"""


def test_edge_case_same_template_different_patient_rejected():
    """THE critical edge case: ~85% identical template, different patients."""
    result = WeightedMedicalDeduplicator().compare(SAME_TEMPLATE_A, SAME_TEMPLATE_B)
    assert result.is_same_patient is False


def test_same_patient_detected_despite_ocr_noise():
    result = WeightedMedicalDeduplicator().compare(DOC_ORIGINAL, DOC_OCR_VARIANT)
    assert result.is_same_patient is True


def test_different_document_rejected():
    result = WeightedMedicalDeduplicator().compare(DOC_ORIGINAL, DOC_DIFFERENT)
    assert result.is_same_patient is False


def test_batch_dedup_three_unique_from_five():
    records = [
        "اسم المريض: أحمد محمد العلي\nرقم المريض: MRN-2026-1007\nالتاريخ: 2026-07-12\nالتشخيص: ارتفاع ضغط الدم",
        "اسم المريض: احمد محمد العلي\nرقم المريض: MRN-2026-1007\nالتاريخ: 2026-07-12\nالتشخيص: ارتفاع ضغط الدم الاساسي",
        "اسم المريض: فاطمة سعيد الحربي\nرقم المريض: MRN-2026-2044\nالتاريخ: 2026-07-12\nالتشخيص: ارتفاع ضغط الدم",
        "اسم المريض: فاطمة سعيد الحربي\nرقم المريض: MRN-2026-2044\nالتاريخ: 2026-07-12\nالتشخيص: ارتفاع ضغط الدم",
        "اسم المريض: سارة عبدالله القحطاني\nرقم المريض: MRN-2026-8899\nالتاريخ: 2026-06-01\nالتشخيص: التهاب الجيوب الأنفية",
    ]
    out = WeightedMedicalDeduplicator().deduplicate(records)
    assert out["unique_count"] == 3
    assert len(out["duplicates"]) == 2


def test_field_aware_similarity_returns_dict_score_one_on_identical():
    s = field_aware_similarity(DOC_ORIGINAL, DOC_ORIGINAL)
    assert isinstance(s, dict)
    assert s["score"] == 1.0
    assert s["is_same_patient"] is True


# ── pipeline smoke ───────────────────────────────────────────────────────────


def test_pipeline_process_pair_smoke():
    pipe = DeduplicationPipeline()
    result = pipe.process_pair(DOC_ORIGINAL, DOC_OCR_VARIANT)
    assert result is not None
    d = result.to_dict()
    assert "steps" in d or "final_similarity" in d or "is_duplicate" in d


def test_pipeline_deduplicate_records_smoke():
    pipe = DeduplicationPipeline()
    out = pipe.deduplicate_records([DOC_ORIGINAL, DOC_OCR_VARIANT, DOC_DIFFERENT])
    assert out is not None
