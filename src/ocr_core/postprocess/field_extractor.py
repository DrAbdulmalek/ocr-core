"""Field extraction helpers for Arabic medical OCR output.

P1-1 hardening (v1.1.0-rc):
- Multi-line value support for diagnosis / medications (captured until the
  next known label or end-of-text).
- Bilingual label patterns (Arabic + English medical shorthand:
  Dx, Pt, DOB, Dr, Rx, etc.).
- Per-field confidence score in [0.0, 1.0] based on label clarity,
  value length, and character class.
- Safe ``build_template_signature()``: replaces values via regex with
  ``re.escape()`` and a unique placeholder, sorted by length descending,
  so a value that is a substring of another does not corrupt the longer
  value (the original ``text.replace(value, ' ')`` had this bug).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re

from ocr_core.postprocess.normalization import arabic_strong_normalize

# ---------------------------------------------------------------------------
# Label patterns — bilingual (Arabic + English)
# ---------------------------------------------------------------------------
# Multi-line fields capture across newlines until the next known label or EOT.
# Single-line fields capture only the rest of the current line.
# Lookahead pattern (shared) stops capture at the next recognized label.
_LABEL_LOOKAHEAD = (
    r"(?=\n\s*(?:"
    r"اسم\s*المريض|المريض|رقم\s*(?:الملف|المريض|الهوية)|"
    r"التاريخ|تاريخ\s*الزيارة|اسم\s*الطبيب|الطبيب|"
    r"التشخيص|الأدوية|العلاج|"
    r"Patient\s*Name|Patient\s*ID|MRN|DOB|Date\b|Doctor|"
    r"Dr\b|Diagnosis|Dx\b|Medications?\b|Rx\b"
    r")\s*[:：\-]?|$)"
)

LABEL_PATTERNS = {
    "patient_name": [
        re.compile(r"(?:اسم\s*المريض|المريض|Patient\s*Name|Pt\s*Name)\s*[:：\-]?\s*(.+)", re.IGNORECASE),
    ],
    "patient_id": [
        re.compile(r"(?:رقم\s*(?:الملف|المريض|الهوية)|Patient\s*ID|MRN|ID|File\s*No)\s*[:：\-]?\s*([A-Z0-9\-/]{3,})", re.IGNORECASE),
    ],
    "date": [
        re.compile(r"(?:التاريخ|تاريخ\s*الزيارة|Date|DOB|Visit\s*Date)\s*[:：\-]?\s*([0-9٠-٩\-/]{6,})", re.IGNORECASE),
    ],
    "doctor_name": [
        re.compile(r"(?:اسم\s*الطبيب|الطبيب|Doctor|Dr)\s*[:：\-]?\s*(.+)", re.IGNORECASE),
    ],
    # Multi-line fields: capture across newlines until next label
    "diagnosis": [
        re.compile(
            r"(?:التشخيص|Diagnosis|Dx)\s*[:：\-]?\s*([\s\S]+?)" + _LABEL_LOOKAHEAD,
            re.IGNORECASE,
        ),
    ],
    "medications": [
        re.compile(
            r"(?:الأدوية|العلاج|Medication[s]?|Rx)\s*[:：\-]?\s*([\s\S]+?)" + _LABEL_LOOKAHEAD,
            re.IGNORECASE,
        ),
    ],
}

# Fields whose values may legitimately span multiple lines.
MULTILINE_FIELDS = frozenset({"diagnosis", "medications"})

MEDICATION_SPLIT_RE = re.compile(r"\s*[،,؛;\n]\s*")
NON_TEMPLATE_VALUE_RE = re.compile(r"\s*[:：\-]\s*.+$")
MULTISPACE_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"[0-9٠-٩]+")

# Unique placeholder used by safe template signature redaction.
# Chosen to never collide with real text (Arabic/Latin/digits).
_VALUE_PLACEHOLDER = "\x00VAL\x00"


def _confidence_for(value: str, field_name: str) -> float:
    """Heuristic confidence in [0.0, 1.0] for an extracted field value.

    Factors:
    - Empty value → 0.0
    - Very short single-token values (<3 chars) → 0.3
    - Longer values get a higher base, capped at 0.95
    - patient_id gets a boost when it matches the expected char class
    - diagnosis/medications get a boost for multi-token richness
    """
    if not value:
        return 0.0
    v = value.strip()
    if not v:
        return 0.0
    tokens = v.split()
    base = min(0.95, 0.5 + 0.05 * len(v))
    if len(v) < 3:
        base = 0.3
    if field_name == "patient_id":
        # IDs should be alnum/dash/slash only — penalize anything else
        if re.fullmatch(r"[A-Z0-9\-/]{3,}", v, re.IGNORECASE):
            base = min(0.95, base + 0.1)
        else:
            base = max(0.2, base - 0.2)
    if field_name in ("diagnosis", "medications"):
        # Multi-token richness = higher confidence
        if len(tokens) >= 2:
            base = min(0.95, base + 0.05 * (len(tokens) - 1))
    return round(base, 3)


@dataclass(slots=True)
class ExtractedMedicalFields:
    patient_name: str = ""
    patient_id: str = ""
    date: str = ""
    doctor_name: str = ""
    diagnosis: str = ""
    medications: list[str] = field(default_factory=list)
    template_signature: str = ""
    raw_text: str = ""
    # P1-1: per-field confidence in [0.0, 1.0].
    # Keys are the same as the scalar fields above; 'medications' is a list
    # of per-item confidences aligned with `medications`.
    confidence: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "patient_name": self.patient_name,
            "patient_id": self.patient_id,
            "date": self.date,
            "doctor_name": self.doctor_name,
            "diagnosis": self.diagnosis,
            "medications": self.medications,
            "template_signature": self.template_signature,
            "raw_text": self.raw_text,
            "fingerprint": self.unique_patient_fingerprint(),
            "confidence": dict(self.confidence),
        }

    def unique_patient_fingerprint(self) -> str:
        base = "|".join([
            self.patient_name.strip(),
            self.patient_id.strip(),
            self.date.strip(),
            self.doctor_name.strip(),
        ])
        return hashlib.sha1(base.encode("utf-8")).hexdigest() if base.strip("|") else ""


class ArabicMedicalFieldExtractor:
    """Regex-first extractor for template-heavy Arabic medical documents."""

    def __init__(self) -> None:
        self.patterns = LABEL_PATTERNS

    @staticmethod
    def _clean_value(value: str) -> str:
        value = MULTISPACE_RE.sub(" ", value).strip(" -:\u200f\u200e")
        return value

    @staticmethod
    def _extract_medications(value: str) -> list[str]:
        parts = [part.strip() for part in MEDICATION_SPLIT_RE.split(value) if part.strip()]
        return parts

    def _first_match(self, field_name: str, text: str) -> str:
        for pattern in self.patterns[field_name]:
            match = pattern.search(text)
            if match:
                return self._clean_value(match.group(1))
        return ""

    def _first_match_raw(self, field_name: str, text: str) -> str:
        """Return raw match group 1 without cleaning (for multi-line fields).

        P1-1: Used for medications/diagnosis so that newlines are preserved
        as separators for downstream splitting.
        """
        for pattern in self.patterns[field_name]:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return ""

    def build_template_signature(self, text: str, fields: ExtractedMedicalFields) -> str:
        """Build a value-redacted template signature from the raw text.

        P1-1: Safe replacement — uses ``re.escape()`` + unique placeholder,
        sorts values by length descending, so a value that is a substring
        of another no longer corrupts the longer value.
        """
        template_lines: list[str] = []
        values_to_remove = [
            fields.patient_name,
            fields.patient_id,
            fields.date,
            fields.doctor_name,
            fields.diagnosis,
            *fields.medications,
        ]
        # Filter empty + deduplicate + sort by length DESC (longest first)
        unique_values = sorted(
            {v for v in values_to_remove if v and v.strip()},
            key=len,
            reverse=True,
        )
        redacted = text
        for value in unique_values:
            # Use re.escape + word-boundary-ish replacement. We can't use \b
            # reliably with Arabic, so we replace all literal occurrences.
            # The placeholder ensures we don't double-redact or corrupt
            # overlapping values (longest-first sort handles containment).
            try:
                redacted = re.sub(re.escape(value), _VALUE_PLACEHOLDER, redacted)
            except re.error:
                # Defensive: if escape produces an invalid pattern, fall back
                # to plain str.replace (the pre-P1 behavior).
                redacted = redacted.replace(value, _VALUE_PLACEHOLDER)
        # Strip placeholders
        redacted = redacted.replace(_VALUE_PLACEHOLDER, " ")
        for raw_line in redacted.splitlines():
            line = NON_TEMPLATE_VALUE_RE.sub("", raw_line).strip()
            line = DIGIT_RE.sub("#", line)
            line = arabic_strong_normalize(line)
            if line:
                template_lines.append(line)
        return "\n".join(template_lines)

    def extract_fields(self, text: str) -> ExtractedMedicalFields:
        raw_text = text or ""
        normalized = arabic_strong_normalize(raw_text)
        fields = ExtractedMedicalFields(raw_text=raw_text)
        fields.patient_name = self._first_match("patient_name", raw_text) or self._first_match("patient_name", normalized)
        fields.patient_id = self._first_match("patient_id", raw_text) or self._first_match("patient_id", normalized)
        fields.date = self._first_match("date", raw_text) or self._first_match("date", normalized)
        fields.doctor_name = self._first_match("doctor_name", raw_text) or self._first_match("doctor_name", normalized)
        # P1-1: diagnosis uses multi-line capture, cleaned at the end
        diag_raw = self._first_match_raw("diagnosis", raw_text) or self._first_match_raw("diagnosis", normalized)
        fields.diagnosis = self._clean_value(diag_raw) if diag_raw else ""
        # P1-1: medications — split on separators BEFORE cleaning (preserves \n)
        meds_raw = self._first_match_raw("medications", raw_text) or self._first_match_raw("medications", normalized)
        if meds_raw:
            fields.medications = [
                self._clean_value(part)
                for part in MEDICATION_SPLIT_RE.split(meds_raw)
                if part.strip()
            ]
        else:
            fields.medications = []
        fields.template_signature = self.build_template_signature(raw_text or normalized, fields)
        # P1-1: per-field confidence scoring
        fields.confidence = {
            "patient_name": _confidence_for(fields.patient_name, "patient_name"),
            "patient_id": _confidence_for(fields.patient_id, "patient_id"),
            "date": _confidence_for(fields.date, "date"),
            "doctor_name": _confidence_for(fields.doctor_name, "doctor_name"),
            "diagnosis": _confidence_for(fields.diagnosis, "diagnosis"),
            "medications": (
                round(
                    sum(_confidence_for(m, "medications") for m in fields.medications) / len(fields.medications),
                    3,
                )
                if fields.medications
                else 0.0
            ),
        }
        # P1-3: instrument decision
        try:
            from app.core.decision_log import log_decision

            extracted_count = sum(
                1 for v in (
                    fields.patient_name, fields.patient_id, fields.date,
                    fields.doctor_name, fields.diagnosis,
                ) if v
            ) + (1 if fields.medications else 0)
            log_decision(
                decision="field_extraction",
                outcome={
                    "extracted_count": extracted_count,
                    "has_medications": bool(fields.medications),
                    "medications_count": len(fields.medications),
                },
                reasons=[
                    f"text_len={len(raw_text)}",
                    f"avg_confidence={sum(fields.confidence.values()) / max(len(fields.confidence), 1):.3f}",
                ],
                inputs={
                    "label_pattern_count": sum(len(p) for p in self.patterns.values()),
                },
                skipped=[] if extracted_count > 0 else ["no_labels_matched"],
            )
        except ImportError:
            pass  # app.core.decision_log not available
        return fields
