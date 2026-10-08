"""normalize_v1 — سياسة تطبيع عربية مُعلنة الإصدار لقياس المعيار الذهبي.

نقل أمين من omni-medical-suite/packages/evaluation/arabic_normalize.py
(قرار معماري A2 حاكم) إلى النواة القانونية ocr-core — S2-T3 / F-13/F-16.

التخزين والقياس بترتيب **منطقي** دائماً. لا ``get_display`` ولا
``arabic_reshaper`` في هذا الملف ولا في أي مسار يقيمه. NFKC إلزامي لمدى
أشكال العرض العربية U+FB50–U+FEFF فقط — لا NFKC عام على كامل النص.

كل خطوة مُعلنة في ``NORMALIZE_V1_STEPS``؛ طيّ الهمزات خيار مُسجّل يُوسم
ناتجه في سجل provenance عند تفعيله — لا قاعدة عمياء.
"""
from __future__ import annotations

import re
import unicodedata

NORMALIZE_V1_VERSION = "normalize_v1/1.0.0"

# أشكال العرض العربية (presentation forms) — NFKC إلزامي لهذا المدى فقط.
_PRESENTATION_FORMS = re.compile(r"[\uFB50-\uFEFF]")
# التشكيل + الألف الخنجرية الصغيرة.
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
_TATWEEL = "\u0640"
# الأرقام العربية-الهندية والممتدة → غربية (توحيد ثنائي الاتجاه في القياس).
_DIGITS_ARABIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_DIGITS_EXTENDED = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
# تطبيع ترقيم أساسي (شكلي فقط — لا حذف ولا تعديل دلالي).
_PUNCT = str.maketrans({"،": ",", "؟": "?", "؛": ";"})
# علامات العرض صفر العرض ومضابط الاتجاه — تُزال لأن الترتيب المنطقي محفوظ أصلاً.
_ZERO_WIDTH = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069]")
_WS = re.compile(r"\s+")

# طيّ الهمزات — اختياري فقط، ولا يُطبق افتراضياً.
_HAMZA_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ؤ": "و", "ئ": "ي"})

NORMALIZE_V1_STEPS = (
    "nfkc_presentation_forms_U+FB50-U+FEFF",
    "strip_zero_width_and_directionals",
    "remove_diacritics_U+064B-U+065F_U+0670",
    "remove_tatweel_U+0640",
    "arabic_indic_digits_to_western",
    "basic_punctuation_unify",
    "collapse_whitespace",
)

OPTIONAL_STEPS = ("fold_hamza",)


def normalize_v1(text: str | None, *, fold_hamza: bool = False):
    """طبّع نصاً عربياً بترتيبه المنطقي، مع سجل سياسة قابل للتسلسل.

    Args:
        text: النص الخام (من OCR أو المرجع).
        fold_hamza: طي متغيرات الهمزة اختيارياً — يُوسم الناتج في
            سجل provenance عند التفعيل.

    Returns:
        Tuple ``(normalized_text, policy)`` حيث ``policy`` قاموس يضم
        الإصدار والخطوات المطبقة وعلم ``changed``.
    """
    if text is None:
        return "", {
            "version": NORMALIZE_V1_VERSION,
            "steps": [],
            "fold_hamza": False,
            "changed": False,
        }
    original = text
    t = _PRESENTATION_FORMS.sub(lambda m: unicodedata.normalize("NFKC", m.group(0)), text)
    t = _ZERO_WIDTH.sub("", t)
    t = _DIACRITICS.sub("", t)
    t = t.replace(_TATWEEL, "")
    t = t.translate(_DIGITS_ARABIC).translate(_DIGITS_EXTENDED)
    t = t.translate(_PUNCT)
    steps = list(NORMALIZE_V1_STEPS)
    if fold_hamza:
        t = t.translate(_HAMZA_FOLD)
        steps.append("fold_hamza")
    t = _WS.sub(" ", t).strip()
    policy = {
        "version": NORMALIZE_V1_VERSION,
        "steps": steps,
        "fold_hamza": fold_hamza,
        "changed": t != original,
    }
    return t, policy
