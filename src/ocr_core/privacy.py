"""وحدة الخصوصية ``ocr_core.privacy`` — إحياء وحدة pii-scrubbing المؤرشفة (OCR-04/#76).

المصدر: ``medical-ocr-training-hub/src/ingestion/hybrid_scrubber.py`` (مستودع مؤرشف)
المستهدف: فحص/تدقيق أي نص مستخرج من تقارير طبية قبل وصوله إلى مكان عام.

قاعدة الوحدة المقصورة (خطة الدمج §8.3 — غير قابلة للتفاوض):
    بيانات المرضى وصور التقارير **أبدًا في git** — HF private/gated datasets
    أو محليًا فقط. لا API خارجي، لا شبكة، لا نماذج تُحمَّل عند الاستيراد.

البنية (كما في الأصل):
  - الطبقة 1: Regex فورية — هواتف (سوريا/السعودية/الإمارات/دولي)، تواريخ
    (ميلادية وأرقام عربية-هندية)، بريد إلكتروني، أرقام هوية وطنية (10–12 رقمًا).
  - الطبقة 2 (اختيارية، تحميل مؤجَّل): NER عربي عبر CamelBERT لأسماء الأشخاص
    والمؤسسات — تُعطَّل برشاقة بغياب transformers أو فشل التحميل، والمكتبة
    لا تنزل أي نموذج وقت الاستيراد ولا في الاختبارات (CI بلا شبكة).

الفروق عن الأصل (موثقة لمراجعة الاستعادة):
  - حذف ``logging.basicConfig`` من الاستيراد (مكتبة لا تُهيئ سجل الجذر).
  - إضافة ``scrub_records`` (سجلات jsonl) و``check_residue`` (فحص بقايا
    بعد التدقيق — يُستخدم في سكربت اختبار السجلات الحقيقية للمالك).
  - إضافة ثوابت السياسة ``PRIVACY_POLICY`` الموثقة في ``docs/PRIVACY.md``.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ثوابت السياسة (القاعدة المقصورة على الوحدة — ocr-core#76 / خطة الدمج §8.3)
# ─────────────────────────────────────────────────────────────────────────────

PRIVACY_POLICY: Dict[str, object] = {
    "rule": "بيانات المرضى وصور التقارير أبدًا في git",
    "allowed_locations": ["local-only", "hf-private", "hf-gated"],
    "forbidden_locations": ["git", "public-hf", "external-api", "cloud-ocr"],
    "reference": "ocr-core#76 (OCR-04) — خطة الدمج §8.3",
    "dataset": "DrAbdulmalek/arabic-medical-ocr-corrections",
    "owner_actions_pending": [
        "gating dataset فورًا (scripts/hf_gate_dataset.py)",
        "مراجعة يدوية موثقة للسجلات قبل أي إعادة نشر",
        "قرار تكلفة crops/ (private dataset مقابل Bucket)",
    ],
}


@dataclass
class RedactionStats:
    """إحصاءات ما حُجب — عدّادات لا محتوى (لا تُطبع نصوص السجلات أبدًا)."""

    phones_found: int = 0
    dates_found: int = 0
    emails_found: int = 0
    ids_found: int = 0
    names_found: int = 0
    organizations_found: int = 0
    total_redactions: int = 0
    residue: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        # كما في الأصل: total يُحسب من العدّادات لحظة الاستدعاء لا مخزَّنًا
        total = (
            self.phones_found + self.dates_found + self.emails_found
            + self.ids_found + self.names_found + self.organizations_found
        )
        return {
            "phones": self.phones_found,
            "dates": self.dates_found,
            "emails": self.emails_found,
            "ids": self.ids_found,
            "names": self.names_found,
            "organizations": self.organizations_found,
            "total": total,
            "residue": dict(self.residue),
        }

    def add(self, other: "RedactionStats") -> None:
        """تجميع إحصاءات دفعة/مجموعة سجلات في هذه الإحصاءات."""
        self.phones_found += other.phones_found
        self.dates_found += other.dates_found
        self.emails_found += other.emails_found
        self.ids_found += other.ids_found
        self.names_found += other.names_found
        self.organizations_found += other.organizations_found
        self.total_redactions += other.total_redactions
        for key, value in other.residue.items():
            self.residue[key] = self.residue.get(key, 0) + value


class HybridPIIScrubber:
    """مسح PII هجين: طبقة Regex فورية + طبقة NER عربية اختيارية (تحميل مؤجَّل).

    ترتيب الأنماط مقصود: التواريخ والهويات والبريد قبل نمط الهاتف العام
    الواسع حتى لا يبتلعها. جرعات الأدوية ورموز الجداول الحيوية محمية من
    الحجب الخاطئ عبر ``PROTECTED_MEDICAL_TERMS``.
    """

    # أنماط الطبقة 1 — الترتيب مهم (التواريخ ثم الهويات ثم البريد ثم الهواتف)
    REGEX_PATTERNS: List[Tuple[re.Pattern, str]] = [
        # --- التواريخ (أولًا لتجنب تعارض نمط الهاتف العام) ---
        # ميلادية: DD/MM/YYYY، DD-MM-YYYY، YYYY-MM-DD...
        (re.compile(r"\b\d{1,2}[-/.\s]\d{1,2}[-/.\s]\d{2,4}\b"), "[REDACTED_DATE]"),
        # أرقام عربية-هندية ٠-٩
        (re.compile(r"[٠-٩]{1,2}[-/.\s][٠-٩]{1,2}[-/.\s][٠-٩]{2,4}"), "[REDACTED_DATE]"),
        # --- الأرقام الوطنية (قبل الهاتف العام) ---
        (re.compile(r"(?<!\d)\d{10,12}(?!\d)"), "[REDACTED_ID]"),
        # --- البريد الإلكتروني ---
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[REDACTED_EMAIL]"),
        # --- أرقام الهواتف (المحددة أولًا ثم العام) ---
        # سوريا: +963-xxx-xxx-xxxx أو 09xx-xxx-xxxx
        (re.compile(r"\+?963[-.\s]?\d{1}[-.\s]?\d{3}[-.\s]?\d{4}"), "[REDACTED_PHONE]"),
        (re.compile(r"0?9\d{1}[-.\s]?\d{3}[-.\s]?\d{4}"), "[REDACTED_PHONE]"),
        # السعودية: +966-5x-xxx-xxxx أو 05x-xxx-xxxx
        (re.compile(r"\+?966[-.\s]?5\d[-.\s]?\d{3}[-.\s]?\d{4}"), "[REDACTED_PHONE]"),
        (re.compile(r"0?5\d[-.\s]?\d{3}[-.\s]?\d{4}"), "[REDACTED_PHONE]"),
        # الإمارات: +971-5x-xxx-xxxx
        (re.compile(r"\+?971[-.\s]?5\d[-.\s]?\d{3}[-.\s]?\d{4}"), "[REDACTED_PHONE]"),
        # دولي عام — يشترط فاصلًا واحدًا على الأقل لتجنب التواريخ والأرقام المجردة
        (re.compile(r"\+?\d{1,4}[-.\s]\(?\d{1,4}\)?[-.\s]\d{1,4}[-.\s]?\d{1,9}"), "[REDACTED_PHONE]"),
    ]

    # مصطلحات طبية تُحمى من الحجب الخاطئ (جرعات تبدو أرقامًا)
    PROTECTED_MEDICAL_TERMS = {
        "500mg", "250mg", "100mg", "50mg", "10mg", "5mg", "1mg",
        "500 mg", "250 mg", "100 mg", "50 mg", "10 mg", "5 mg", "1 mg",
        "BP", "HR", "RR", "SpO2", "BMI",
    }

    _REPLACEMENT_TO_STAT = {
        "[REDACTED_PHONE]": "phones_found",
        "[REDACTED_DATE]": "dates_found",
        "[REDACTED_EMAIL]": "emails_found",
        "[REDACTED_ID]": "ids_found",
    }

    def __init__(self, use_ner: bool = True, ner_model: Optional[str] = None):
        """تهيئة المساح.

        Args:
            use_ner: تفعيل طبقة NER لأسماء الأشخاص والمؤسسات (أدق، أبطأ،
                تحميل مؤجَّل عند أول استخدام فعلي).
            ner_model: اسم نموذج HF لـNER (الافتراضي CamelBERT MSA NER).
        """
        self.use_ner = use_ner
        self.ner_model_name = ner_model or "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"
        self._ner_pipeline = None
        self._ner_loaded = False
        self._compiled_patterns = list(self.REGEX_PATTERNS)

    # ── الطبقة 2: NER (تحميل مؤجَّل — لا شيء يُنزل عند الاستيراد) ──

    def _load_ner_model(self) -> None:
        """تحميل النموذج عند أول حاجة فقط، مع تعطيل رشيق عند الفشل."""
        if self._ner_loaded:
            return
        try:
            from transformers import pipeline

            logger.info("تحميل نموذج NER: %s ...", self.ner_model_name)
            self._ner_pipeline = pipeline(
                "ner",
                model=self.ner_model_name,
                aggregation_strategy="simple",
            )
            self._ner_loaded = True
            logger.info("نموذج NER جاهز")
        except ImportError:
            logger.warning("transformers غير مثبتة — NER معطل. التثبيت: pip install transformers")
            self.use_ner = False
        except Exception as exc:  # فشل تحميل النموذج (شبكة/ذاكرة) → regex فقط
            logger.warning("فشل تحميل NER: %s — التراجع إلى وضع Regex فقط", exc)
            self.use_ner = False

    # ── الواجهة الرئيسية ──

    def scrub(self, text: str) -> Tuple[str, RedactionStats]:
        """مسح PII من نص: طبقة Regex ثم NER إن كانت مفعلة.

        العائد: (النص بعد الحجب، الإحصاءات).
        """
        stats = RedactionStats()
        redacted = self._regex_scrub(text, stats)
        if self.use_ner:
            redacted = self._ner_scrub(redacted, stats)
        stats.total_redactions = (
            stats.phones_found + stats.dates_found + stats.emails_found
            + stats.ids_found + stats.names_found + stats.organizations_found
        )
        return redacted, stats

    def check_residue(self, text: str) -> Dict[str, int]:
        """فحص بقايا PII في نص (بعد التدقيق أو أصلاً) — عدّادات بلا محتوى.

        يُستخدم في ``scripts/privacy_selftest.py`` لإثبات نظافة السجلات
        قبل أي إعادة نشر؛ العائد: تصنيف → عدد المطابقات المتبقية.
        """
        residue: Dict[str, int] = {}
        for pattern, replacement in self._compiled_patterns:
            hits = len(pattern.findall(text))
            if hits:
                label = self._REPLACEMENT_TO_STAT.get(replacement, replacement)
                residue[label] = residue.get(label, 0) + hits
        return residue

    def scrub_records(
        self,
        records: Iterable[dict],
        text_keys: Sequence[str] = ("text", "ocr_text", "content"),
        check: bool = True,
    ) -> Tuple[List[dict], RedactionStats]:
        """مسح حقول النص في سجلات jsonl (مثل تصحيحات arabic-medical-ocr-corrections).

        الحقول غير النصية تبقى كما هي حرفيًا؛ مع ``check=True`` يُحسب فحص
        البقايا لكل سجل بعد التدقيق ويُجمَّع في ``stats.residue``.

        العائد: (السجلات بعد الحجب، الإحصاءات المجمعة).
        """
        keys = [k for k in text_keys if k]
        scrubbed: List[dict] = []
        total = RedactionStats()
        for record in records:
            if not isinstance(record, dict):
                scrubbed.append(record)
                continue
            out = dict(record)
            for key in keys:
                value = out.get(key)
                if isinstance(value, str) and value:
                    redacted, stats = self.scrub(value)
                    out[key] = redacted
                    total.add(stats)
            if check:
                residue_text = " ".join(
                    out[k] for k in keys if isinstance(out.get(k), str)
                )
                for label, count in self.check_residue(residue_text).items():
                    total.residue[label] = total.residue.get(label, 0) + count
            scrubbed.append(out)
        return scrubbed, total

    # ── الطبقات الداخلية ──

    def _regex_scrub(self, text: str, stats: RedactionStats) -> str:
        """الطبقة 1: حجب سريع بالأنماط — عدّادات دقيقة بلا محتوى."""
        for pattern, replacement in self._compiled_patterns:
            matches = pattern.findall(text)
            text = pattern.sub(replacement, text)
            stat_key = self._REPLACEMENT_TO_STAT.get(replacement)
            if stat_key and matches:
                setattr(stats, stat_key, getattr(stats, stat_key) + len(matches))
        return text

    def _ner_scrub(self, text: str, stats: RedactionStats) -> str:
        """الطبقة 2: كيانات NER (أسماء/مؤسسات) — استبدال من اليمين لليسار."""
        self._load_ner_model()
        if not self._ner_pipeline:
            return text
        try:
            entities = self._ner_pipeline(text)
            for entity in reversed(entities):
                entity_group = entity.get("entity_group", "")
                if entity_group in ("PER", "B-PER", "I-PER"):
                    entity_text = text[entity["start"]:entity["end"]]
                    if entity_text.strip() not in self.PROTECTED_MEDICAL_TERMS:
                        text = text[:entity["start"]] + "[REDACTED_NAME]" + text[entity["end"]:]
                        stats.names_found += 1
                elif entity_group in ("ORG", "B-ORG", "I-ORG"):
                    text = text[:entity["start"]] + "[REDACTED_ORG]" + text[entity["end"]:]
                    stats.organizations_found += 1
        except Exception as exc:
            logger.error("فشل مسح NER: %s", exc)
        return text

    def scrub_batch(self, texts: List[str]) -> List[Tuple[str, RedactionStats]]:
        """مسح عدة نصوص — عائد قائمة (نص، إحصاءات)."""
        return [self.scrub(text) for text in texts]


def scrub_pii(text: str, use_ner: bool = True) -> Tuple[str, RedactionStats]:
    """دالة راحة متوافقة مع واجهة الأصل (v1/v2) — مسح نص واحد."""
    return HybridPIIScrubber(use_ner=use_ner).scrub(text)
