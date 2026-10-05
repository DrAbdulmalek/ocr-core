# src/ocr_core/postprocess/matching_view.py
"""عرض مطابقة غير تدميري للنص العربي — التزامًا مع ميثاق الدليل البصري.

الفكرة (Clean Room Design — أصلية 100%):
    أنشطة المعالجة اللاحقة مثل ``normalization.arabic_normalize`` تُعيد نصًا
    مُطبَّعًا وتُهمل النص الأصلي. هذا الوحدة تُقدّم **عقدًا غير تدميري**:
    النص الأصلي يبقى كما هو حرفيًا، والوظيفة تُعيد «نسخة مطابقة» منفصلة
    تُستخدم حصرًا في المقارنة/الاسترجاع/الربط، مع سجل تدقيق لكل قاعدة
    تطبيق وحصيلته من التغييرات.

ما أُعيد تنفيذه بفعالية (وكان ناقصًا في مسودات خارجية):
    - حماية المقادير الطبية **فعليًا** وليست شكلية: كل رمز «رقم + وحدة»
      (بالأرقام الغربية أو العربية-الهندية، وبالوحدات اللاتينية أو العربية)
      يُقنَّع قبل تطبيق القواعد ويُستعاد حرفيًا بعدها — جرعات مثل
      ``500 mg`` أو ``٥٠ ملغ`` أو ``36.7 °C`` لا تمسّها أي قاعدة تطبيع.
    - توحيد التاء المربوطة (ة → ه) **اختياري** افتراضيًا، إشارةً إلى تحذير
      انحياز المقاييس الموثق في ``normalization.arabic_strong_normalize``:
      تطبيقه على المرجع والتنبؤ معًا يخفي أخطاء ة/ه الحقيقية، وبعض
      المصطلحات السريرية يتغير معناها بين ة و ه.

اتساق رياضي:
    أصناف المحارف هنا مطابقة لأصناف ``normalization.py`` (التشكيل
    ``[\u0617-\u061A\u064B-\u0652]``، الألف ``[إأٱآ]``، التطويل، الأرقام
    العربية-الهندية) حتى لا تتباعد نتائج المطابقة عن خط الأنابيب الرئيسي.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

__all__ = ["MatchingChange", "MatchingResult", "to_matching_view"]

# ── القواعد ──────────────────────────────────────────────────────────────────
# أصناف المحارف متوافقة مع normalization.py عمدًا (انظر docstring الوحدة).

_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
_ALEF_RE = re.compile(r"[\u0623\u0625\u0622\u0671]")  # أ إ آ ٱ → ا
_ALEF_MAQSURA_RE = re.compile(r"\u0649")  # ى → ي
_TAA_MARBUTA_RE = re.compile(r"\u0629")  # ة → ه (اختياري — انظر التحذير)
_TATWEEL_RE = re.compile(r"\u0640+")
_ARABIC_DIGIT_RE = re.compile(r"[\u0660-\u0669]")
_WHITESPACE_RE = re.compile(r"\s+")


def _arabic_digit_to_western(match: re.Match) -> str:
    return str(int(match.group(0), 10))


@dataclass(frozen=True)
class _Rule:
    name: str
    pattern: re.Pattern
    replacement: object  # نص أو دالة استبدال
    note: str = ""


_RULES: tuple[_Rule, ...] = (
    _Rule("remove_diacritics", _DIACRITICS_RE, ""),
    _Rule("unify_alef", _ALEF_RE, "ا"),
    _Rule("unify_alef_maqsura", _ALEF_MAQSURA_RE, "ي"),
    _Rule("remove_tatweel", _TATWEEL_RE, ""),
    _Rule("unify_arabic_digits", _ARABIC_DIGIT_RE, _arabic_digit_to_western),
    _Rule("collapse_whitespace", _WHITESPACE_RE, " ", note="يطبق بعد الاستعادة كي لا يفسد الأقنعة"),
    _Rule(
        "unify_taa_marbuta",
        _TAA_MARBUTA_RE,
        "ه",
        note="اختياري: يخفي أخطاء ة/ه من CER/WER وقد يغير معنى مصطلحات سريرية",
    ),
)

# ── حماية المقادير الطبية ────────────────────────────────────────────────────

# رقم (غربي أو عربي-هندي، مع فاصلة عشرية/ألف اختيارية) + مسافة اختيارية + وحدة.
_UNITS_LATIN = r"(?:mg|mcg|ug|ml|iu|g|kg|mm|cm|mmol|°C|%)"
_UNITS_ARABIC = r"(?:ملغ|ميكروغرام|مكغ|مل|جم|كجم|غرام|كيلوغرام|سم|مم|مليمول|وحدة دولية)"
_MEDICAL_TOKEN_RE = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)?|[\u0660-\u0669]+(?:[.,][\u0660-\u0669]+)?)[ \u00a0]?"
    rf"(?:{_UNITS_LATIN}|{_UNITS_ARABIC})(?![\w])",
    re.IGNORECASE,
)

_MASK = "\x00{}\x00"  # لن يلمسها أي صنف محارف في القواعد أعلاه

_RULES_BY_NAME = {rule.name: rule for rule in _RULES}


def _mask_medical(text: str) -> tuple[str, list[str]]:
    """إخفاء المقادير الطبية بمعزولات آمنة وإرجاع (النص المقنّع، القيم الأصلية)."""
    protected: list[str] = []

    def _sub(match: re.Match) -> str:
        protected.append(match.group(0))
        return _MASK.format(len(protected) - 1)

    return _MEDICAL_TOKEN_RE.sub(_sub, text), protected


def _unmask(text: str, protected: list[str]) -> str:
    """استعادة المقادير المحمية حرفيًا كما وردت."""
    for index, value in enumerate(protected):
        text = text.replace(_MASK.format(index), value)
    return text


# ── الواجهة العامة ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MatchingChange:
    """تغيير واحد في سجل التدقيق: اسم القاعدة وعدد مواضع تطبيقها."""

    rule: str
    count: int
    note: str = ""


@dataclass(frozen=True)
class MatchingResult:
    """نتيجة غير تدميرية: الأصل محفوظ حرفيًا + نسخة للمطابقة + سجل تغييرات."""

    original_text: str
    matching_text: str
    changes: tuple[MatchingChange, ...] = field(default_factory=tuple)

    def change_counts(self) -> dict[str, int]:
        """خريطة اسم القاعدة → عدد التغييرات (للتقارير والمراقبة)."""
        return {change.rule: change.count for change in self.changes}


def to_matching_view(
    text: str,
    *,
    taa_marbuta: bool = False,
    protect_medical: bool = True,
) -> MatchingResult:
    """بناء «عرض مطابقة» للنص دون المساس بالأصل.

    Args:
        text: نص OCR الأصلي (يُحفظ حرفيًا في النتيجة).
        taa_marbuta: تفعيل توحيد ة → ه (اختياري — انظر تحذير انحياز المقاييس).
        protect_medical: إخفاء المقادير الطبية عن كل القواعد واستعادتها حرفيًا.

    Returns:
        MatchingResult يحمل النص الأصلي، نسخة المطابقة، وسجل التغييرات.
    """
    if not text:
        return MatchingResult(original_text=text or "", matching_text="", changes=())

    original = text
    working = unicodedata.normalize("NFC", text)

    masked, protected = (
        _mask_medical(working) if protect_medical else (working, [])
    )

    counts: dict[str, int] = {}
    notes: dict[str, str] = {}
    for rule in _RULES:
        if rule.name == "unify_taa_marbuta" and not taa_marbuta:
            continue  # توحيد التاء اختياري افتراضيًا (تحذير انحياز المقاييس)
        if rule.name == "collapse_whitespace":
            continue  # يُحصى بعد الاستعادة أدناه
        new_text, count = rule.pattern.subn(rule.replacement, masked)
        if count:
            counts[rule.name] = count
            notes[rule.name] = rule.note
            masked = new_text

    restored = _unmask(masked, protected)
    collapsed = _WHITESPACE_RE.sub(" ", restored).strip()
    if collapsed != restored:
        counts["collapse_whitespace"] = 1
        notes["collapse_whitespace"] = _RULES_BY_NAME["collapse_whitespace"].note
    restored = collapsed

    changes = tuple(
        MatchingChange(rule=name, count=count, note=notes.get(name, ""))
        for name, count in counts.items()
    )
    # النسخة المطابقة تعكس النص بعد القواعد؛ الأصل يبقى حرفيًا مهما حدث.
    return MatchingResult(original_text=original, matching_text=restored, changes=changes)
