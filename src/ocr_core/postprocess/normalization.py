# src/ocr/normalization.py
"""
Advanced Arabic Text Normalization for Medical OCR.
Includes: Unicode NFC, diacritics removal, character unification,
Arabic digit conversion, medical dictionary mapping, and optional CAMeL Tools integration.
"""
import json
import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Medical Dictionary (mutable, loaded from JSON) ──────────────────────────
MEDICAL_DICT: dict[str, str] = {
    # Abbreviation expansions (common in prescriptions)
    "مغ": "ملغ",
    "سم": "سم",
    "ص": "صباحا",
    "م": "مساء",
    "Name": "الاسم",
    # Drug forms
    "قرص": "قرص",
    "حبة": "حبة",
    "كبسولة": "كبسولة",
    "شراب": "شراب",
    "حقن": "حقن",
    "مرهم": "مرهم",
    "قطرة": "قطرة",
    "بخاخ": "بخاخ",
    "محلول": "محلول",
    # Common medical terms
    "تشخيص": "تشخيص",
    "التهاب": "التهاب",
    "لوزتين": "اللوزتين",
    # Drug name corrections
    "اموكسيسيلين": "أموكسيسيلين",
    "ايبوبروفين": "إيبوبروفين",
    "باراسيتامول": "باراسيتامول",
    "اموكلاف": "أموكسيسيلين/كلافولانات",
}


def load_medical_dict(dict_path: str | None = None) -> None:
    """Load additional medical dictionary from JSON file and merge into MEDICAL_DICT."""
    global MEDICAL_DICT
    if dict_path is None:
        dict_path = str(Path(__file__).parent / "data" / "medical_terms.json")  # packaged data (deviation: omni used repo-root path)
    p = Path(dict_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                MEDICAL_DICT.update(loaded)
                logger.info(f"Loaded {len(loaded)} terms from {dict_path}")


# ── Core Normalization ──────────────────────────────────────────────────────

# Pre-compiled patterns for performance
_DIACRITICS_RE = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
_ALEF_RE = re.compile(r'[إأٱآ]')
_TATWEEL_RE = re.compile(r'[ـ]+')
_WHITESPACE_RE = re.compile(r'\s+')
_ARABIC_DIGIT_RE = re.compile(r'[٠-٩]')
_CLEAN_RE = re.compile(r'[^\w\s\.\-\،\؛\؟\:\(\)]')


def arabic_normalize(text: str) -> str:
    """
    Basic Arabic normalization for medical OCR text.
    Steps: NFC → remove diacritics → unify alef/ya/teh → remove tatweel →
           normalize digits → clean special chars.
    """
    if not text:
        return ""
    # 1. Unicode NFC
    text = unicodedata.normalize('NFC', text)
    # 2. Remove diacritics (tashkeel)
    text = _DIACRITICS_RE.sub('', text)
    # 3. Unify hamza/alef variants
    text = _ALEF_RE.sub('ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    # 4. Clean spaces and tatweel
    text = _TATWEEL_RE.sub('', text)
    text = _WHITESPACE_RE.sub(' ', text).strip()
    # 5. Convert Arabic-Indic digits to Western
    text = _ARABIC_DIGIT_RE.sub(lambda m: str(int(m.group(0), 10)), text)
    # 6. Remove unwanted special characters (keep medical-relevant ones)
    text = re.sub(r'[^\w\s\.\-\،\؛]', '', text)
    return text.strip()


def arabic_strong_normalize(text: str, use_medical_dict: bool = True) -> str:
    """
    Advanced Arabic normalization with medical dictionary mapping.
    Includes all steps of arabic_normalize() plus optional medical term correction.

    Args:
        text: Input Arabic medical text (from OCR).
        use_medical_dict: If True, apply medical dictionary corrections.

    Returns:
        Normalized text with medical terms corrected.

    ⚠️ METRIC BIAS WARNING:
        This function converts ة → ه (line below), which is applied to BOTH
        the prediction AND the reference inside compute_metrics. While this
        is methodologically sound (apples-to-apples comparison), it HIDES
        real ة/ه OCR errors from CER/WER scores. In a medical context this
        matters: "صيدلية" (pharmacy) vs "صيدليه" changes grammatical role,
        and some clinical terms have distinct meanings with ة vs ه.

        To see true ة/h error rates, run a separate evaluation WITHOUT
        calling this function, or count ة/ه mismatches separately.
    """
    if not text:
        return ""
    # 1. Unicode NFC
    text = unicodedata.normalize('NFC', text)
    # 2. Remove diacritics
    text = _DIACRITICS_RE.sub('', text)
    # 3. Unify characters
    text = _ALEF_RE.sub('ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = _TATWEEL_RE.sub('', text)
    # 4. Clean spaces
    text = _WHITESPACE_RE.sub(' ', text).strip()
    # 5. Convert Arabic digits
    text = _ARABIC_DIGIT_RE.sub(lambda m: str(int(m.group(0), 10)), text)
    # 6. Medical Dictionary Mapping
    if use_medical_dict:
        words = text.split()
        normalized_words = []
        for word in words:
            lower = word.lower()
            if lower in MEDICAL_DICT:
                normalized_words.append(MEDICAL_DICT[lower])
            else:
                normalized_words.append(word)
        text = ' '.join(normalized_words)
    # 7. Final cleanup (keep more medical-relevant chars)
    text = _CLEAN_RE.sub('', text)
    return text.strip()


def normalize_batch(texts: list, use_medical_dict: bool = True) -> list:
    """Apply arabic_strong_normalize to a list of texts."""
    return [arabic_strong_normalize(t, use_medical_dict) for t in texts]


def full_arabic_normalize(text: str) -> str:
    """
    Full normalization pipeline combining CAMeL Tools (if available)
    with our custom medical normalization.

    Falls back to arabic_strong_normalize if CAMeL is not installed.
    """
    try:
        from camel_tools.utils.normalize import normalize_unicode
        text = normalize_unicode(text)
    except ImportError:
        logger.debug("CAMeL Tools not available, using basic NFC")
        text = unicodedata.normalize('NFC', text)
    return arabic_strong_normalize(text)


def _fix_digit_letter_confusion(text: str) -> str:
    """Fix common OCR confusions between Arabic letters and digits/look-alikes.

    Medical OCR frequently misreads:
      - Arabic 'ح' (haa) <-> digit '7'
      - Arabic 'د' (daal) <-> digit '3' (or Persian numeral)
      - Arabic 'ط' (taa) <-> digit '6'
      - Arabic 'ه' (haa) <-> digit '5'
      - Arabic 'و' (waaw) <-> digit '9'
      - Arabic 'ب' (baa) <-> digit '3' (context: mg)
      - Latin 'O' <-> digit '0'
      - Latin 'l' <-> digit '1'
      - Latin 'S' <-> digit '5'
      - Latin 'B' <-> digit '8'

    Uses surrounding medical context to decide the correct character.
    """
    if not text:
        return text

    confusion_rules = [
        # "mg" misread as "m7" (Arabic haa looks like 7)
        (r'\bm[757]\b', 'ملغ'),
        (r'\b[757]g\b', 'ملغ'),
        # "500 mg" with Arabic mim or Latin m
        (r'(\d+)\s*[مm][757]', r'\1 ملغ'),
        (r'(\d{2,4})\s*[مm][757]', r'\1 ملغ'),
        # "ml" misread as "m1" or "m|"
        (r'(\d+)\s*m[l1]', r'\1 مل'),
        # "cm" with O/0 confusion
        (r'(\d+)\s*c[0oO]m(?!\w)', r'\1 سم'),
        # "mm" with Arabic mim
        (r'(\d+)\s*m[مm]{2}(?!\w)', r'\1 مم'),
        # Temperature: "37C" variants (Arabic sin or Latin C)
        (r'3[77][\.\s]*[cCس]', '37 درجة مئوية'),
        # Latin O/0 confusion — token-level replacement (not single-char
        # surrounded by digits). Old `(?<=\d)O(?=\d)` failed on consecutive
        # letters like "5OO" (first O not followed by digit, second not
        # preceded by digit). Also handles "5OOmg" where \b fails between
        # O and m (both word chars), so we use (?![0-9Oo]) instead of \b.
        (r'\b(?=[0-9oO]*\d[0-9oO]*(?![0-9Oo]))[0-9oO]+(?![0-9Oo])',
         lambda m: m.group().replace('O', '0').replace('o', '0')),
        # "IV" (intravenous) confused with "1V"
        (r'\b1V\b', 'IV'),
        # "tab" confused with "7ab"
        (r'\b7ab(?!\w)', 'tab'),
    ]

    result = text
    for pattern, replacement in confusion_rules:
        result = re.sub(pattern, replacement, result)
    return result


# ── Load medical dict on import ─────────────────────────────────────────────
try:
    load_medical_dict()
except Exception as e:
    logger.warning(f"Could not load medical_terms.json: {e}")


# ── CLI Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_texts = [
        "الاسم: أحمد محمد - التشخيص: إلتهاب اللوزتين",
        "المريض يأخذ اموكسيسيلين 500مغ صباحا ومساء",
        "وصفة: ٢ قرص باراسيتامول ٥٠٠ ملغ",
    ]
    for t in test_texts:
        print(f"Original : {t}")
        print(f"Basic    : {arabic_normalize(t)}")
        print(f"Strong   : {arabic_strong_normalize(t)}")
        print(f"Full     : {full_arabic_normalize(t)}")
        print("---")
