"""Arabic RTL fixer for OCR output — ported from omni-medical-suite/src/ocr/rtl_utils.py.

Adaptations to ocr-core (documented, no silent behavior drift):
- ``packages.nlp.arabic_rtl.ARABIC_NORMALIZATION_MAP`` (omni) is replaced by a map
  BUILT AT IMPORT TIME from :mod:`unicodedata` (NFKC) — single source of truth,
  structurally immune to the hand-written off-by-2 drift found in omni's map
  (FEC7/FEC8 Zah mapped to Ain, FEDD/FEDE Lam mapped to Kaf, ...).
- The optional ``packages.vision.text_reconstructor.canonicalize_arabic`` bridge is
  replaced by the same NFKC+map local implementation (the bridge agreed with it by contract).
- ``app.core.decision_log`` instrumentation removed (logger.debug instead).

The fixer addresses the EasyOCR failure mode from the July 2026 validation pass:
Arabic spans may be emitted visually left-to-right with each Arabic token reversed.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Arabic + English messages (parity with omni)
_MSG_NOT_ARABIC = "لا يوجد نص عربي | no Arabic content"
_MSG_FIXING = "إصلاح اتجاه العربية | fixing Arabic RTL order"


def _build_presentation_form_map() -> dict[str, str]:
    """Build the presentation-form → base-letter map from unicodedata itself.

    Covers U+FE81..U+FEFC where NFKC yields standard Arabic letters (0621-064A),
    including the lam-alef ligatures (two-char NFKC results). Cannot drift:
    every entry is derived, never hand-written.
    """
    arabic_letters = {chr(c) for c in range(0x0621, 0x064B)}
    out: dict[str, str] = {}
    for cp in range(0xFE81, 0xFEFD):
        ch = chr(cp)
        nfkc = unicodedata.normalize("NFKC", ch)
        if all(c in arabic_letters for c in nfkc):
            out[ch] = nfkc
    return out


ARABIC_NORMALIZATION_MAP: dict[str, str] = _build_presentation_form_map()

ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
ARABIC_TOKEN_RE = re.compile(r"^[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+$")
PRESENTATION_FORM_RE = re.compile(r"[\uFB50-\uFDFF\uFE70-\uFEFF]")
TOKEN_SPLIT_RE = re.compile(r"\s+")
COMMON_ARABIC_HINTS = (
    "ال",
    "اسم",
    "الم",
    "مريض",
    "تاريخ",
    "رقم",
    "تشخيص",
    "دواء",
    "عبد",
    "بن",
    "ية",
    "ات",
)


@dataclass(slots=True)
class RTLFixStats:
    """Diagnostics returned by :class:`ArabicRTLFixer`."""

    reversal_ratio: float
    had_presentation_forms: bool
    changed: bool


class ArabicRTLFixer:
    """Fix reversed Arabic OCR lines while leaving non-Arabic tokens intact."""

    def __init__(self, reversal_threshold: float = 0.30) -> None:
        self.reversal_threshold = reversal_threshold

    @staticmethod
    def contains_arabic(text: str) -> bool:
        return bool(text and ARABIC_CHAR_RE.search(text))

    @staticmethod
    def normalize_presentation_forms(text: str) -> str:
        """Normalize presentation forms to canonical Unicode (NFKC + derived map)."""
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        return "".join(ARABIC_NORMALIZATION_MAP.get(ch, ch) for ch in normalized)

    @staticmethod
    def _arabic_tokens(text: str) -> list[str]:
        return [token for token in TOKEN_SPLIT_RE.split(text.strip()) if ARABIC_TOKEN_RE.match(token)]

    @staticmethod
    def _arabic_hint_score(token: str) -> int:
        if not token:
            return 0
        score = 0
        if token.startswith("ال"):
            score += 2
        for hint in COMMON_ARABIC_HINTS:
            if hint in token:
                score += 1
        return score

    def reversal_ratio(self, text: str) -> float:
        """Estimate how strongly the text looks like visually reversed Arabic.

        Heuristic: if reversing a token yields more common Arabic prefixes and
        substrings than the token in its current form, that token votes for a
        reversal fix.
        """
        tokens = self._arabic_tokens(self.normalize_presentation_forms(text))
        long_tokens = [token for token in tokens if len(token) >= 3]
        if not long_tokens:
            return 0.0

        reversed_votes = 0
        for token in long_tokens:
            current_score = self._arabic_hint_score(token)
            reversed_score = self._arabic_hint_score(token[::-1])
            if reversed_score > current_score:
                reversed_votes += 1
        return reversed_votes / len(long_tokens)

    def should_fix(self, text: str) -> bool:
        if not self.contains_arabic(text):
            logger.debug(_MSG_NOT_ARABIC)
            return False
        normalized = self.normalize_presentation_forms(text)
        if PRESENTATION_FORM_RE.search(text):
            return True
        return self.reversal_ratio(normalized) >= self.reversal_threshold

    @staticmethod
    def _reverse_arabic_token(token: str) -> str:
        return token[::-1]

    def _fix_line(self, line: str) -> str:
        tokens = [token for token in TOKEN_SPLIT_RE.split(line.strip()) if token]
        if not tokens:
            return ""

        normalized = [self.normalize_presentation_forms(token) for token in tokens]
        converted = [
            self._reverse_arabic_token(token) if ARABIC_TOKEN_RE.match(token) else token
            for token in normalized
        ]

        arabic_positions = [idx for idx, token in enumerate(normalized) if ARABIC_TOKEN_RE.match(token)]
        if len(arabic_positions) > 1:
            reversed_arabic = [converted[idx] for idx in arabic_positions][::-1]
            for idx, new_token in zip(arabic_positions, reversed_arabic, strict=False):
                converted[idx] = new_token
        return " ".join(converted)

    def fix_text(self, text: str, *, force: bool = False) -> str:
        if not text:
            return ""
        normalized = self.normalize_presentation_forms(text)
        if not force and not self.should_fix(normalized):
            return normalized
        logger.debug(_MSG_FIXING)
        lines = [self._fix_line(line) for line in normalized.splitlines()]
        return "\n".join(lines).strip()

    def analyze_and_fix(self, text: str, *, force: bool = False) -> tuple[str, RTLFixStats]:
        normalized = self.normalize_presentation_forms(text)
        ratio = self.reversal_ratio(normalized)
        changed = force or self.should_fix(normalized)
        fixed = self.fix_text(normalized, force=force)
        stats = RTLFixStats(
            reversal_ratio=ratio,
            had_presentation_forms=bool(PRESENTATION_FORM_RE.search(text or "")),
            changed=changed and fixed != normalized,
        )
        return fixed, stats
