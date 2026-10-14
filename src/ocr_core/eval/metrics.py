"""مقاييس تقييم دقة OCR — النسخة القانونية في النواة (F-13/F-16).

منقولة أميناً من omni-medical-suite/packages/evaluation/metrics.py
(المصدر الأصلي: دمج مشروع advanced-ocr) مع الإضافة الحاكمة الوحيدة:
``skip_internal_normalize`` — التي تمنع **التطبيع المزدوج** عندما يكون
المستدعي طبّع الطرفين مسبقاً عبر ``ocr_core.eval.normalize_v1``.

العقد:
    calculate_cer(reference, hypothesis, *, skip_internal_normalize=False)
    calculate_wer(reference, hypothesis, *, skip_internal_normalize=False)

- skip_internal_normalize=False (افتراضي): تُطبَّع المدخلات داخلياً عبر
  ``_normalize_arabic`` — سلوك متوافق تماماً مع النسخة السابقة.
- skip_internal_normalize=True: تُستخدم المدخلات كما هي — هذا هو المسار
  الإلزامي لأي رقم CER/WER منشور: طبّع الطرفين عبر ``normalize_v1``
  (نقطة تطبيع واحدة) ثم استدعِ بـ True. أي رقم قبله غير موثوق (F-13/F-16).
"""
from __future__ import annotations

import re

# =========================================================================
# تطبيع داخلي خفيف (سلوك التوافق — نفس دلالات omni/metrics.py حرفياً)
# =========================================================================

def _normalize_arabic(text: str | None) -> str:
    """تطبيع النص العربي للمقارنة: إزالة التشكيل، توحيد الألف/الياء، المسافات."""
    if text is None:
        return ""
    if not text:
        return ""
    # إزالة التشكيل العربي
    diacritics = r"[\u064B-\u065F\u0670]"
    text = re.sub(diacritics, "", text)
    # توحيد أشكال الألف
    text = text.replace("إ", "ا").replace("أ", "ا").replace("ٱ", "ا")
    # توحيد الألف المقصورة
    text = text.replace("ى", "ي")
    # توحيد المسافات
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================================
# Levenshtein (نفس الخوارزمية والتعليقات — إرجاع (التعديلات، 0، 0))
# =========================================================================

def _levenshtein_distance(s1, s2) -> tuple[int, int, int]:
    """مسافة Levenshtein. تعيد (إجمالي_التعديلات، استبدالات، إدراجات_وحذف)."""
    len1, len2 = len(s1), len(s2)

    if len1 == 0:
        return (len2, 0, len2)
    if len2 == 0:
        return (len1, 0, len1)

    d = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        d[i][0] = i
    for j in range(len2 + 1):
        d[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # حذف / Deletion
                d[i][j - 1] + 1,        # إدراج / Insertion
                d[i - 1][j - 1] + cost,  # استبدال / Substitution
            )

    return (d[len1][len2], 0, 0)


# =========================================================================
# المقاييس — مع skip_internal_normalize (F-13/F-16)
# =========================================================================

def calculate_cer(
    reference: str, hypothesis: str, *, skip_internal_normalize: bool = False
) -> tuple[float, int, int]:
    """معدل خطأ الأحرف. CER = (S + D + I) / N حيث N أحرف المرجع.

    Returns:
        (cer, أخطاء, إجمالي_أحرف_المرجع)
    """
    if not skip_internal_normalize:
        reference = _normalize_arabic(reference)
        hypothesis = _normalize_arabic(hypothesis)

    if not reference:
        return (0.0 if not hypothesis else 1.0, len(hypothesis), 0)

    edits, _, _ = _levenshtein_distance(reference, hypothesis)
    cer = edits / len(reference)
    return (cer, edits, len(reference))


def calculate_wer(
    reference: str, hypothesis: str, *, skip_internal_normalize: bool = False
) -> tuple[float, int, int]:
    """معدل خطأ الكلمات. WER = (S + D + I) / N حيث N كلمات المرجع.

    Returns:
        (wer, أخطاء, إجمالي_كلمات_المرجع)
    """
    if not skip_internal_normalize:
        reference = _normalize_arabic(reference)
        hypothesis = _normalize_arabic(hypothesis)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        return (0.0 if not hyp_words else 1.0, len(hyp_words), 0)

    edits, _, _ = _levenshtein_distance(ref_words, hyp_words)
    wer = edits / len(ref_words)
    return (wer, edits, len(ref_words))
