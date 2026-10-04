"""
metrics.py — OCR quality metrics calculation.
حساب مقاييس جودة التعرف البصري.

Provides:
- CER (Character Error Rate)
- WER (Word Error Rate)
- Medical Term Accuracy
- Combined metrics calculation
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import rapidfuzz

# Medical term dictionaries — Arabic and English
ENGLISH_MEDICAL_TERMS: set = {
    # Cardiology
    "angina", "pectoris", "cardiac", "cardiology", "myocardial", "infarction",
    "arrhythmia", "fibrillation", "tachycardia", "bradycardia", "cardiomyopathy", "hypertension",
    "electrocardiogram", "echocardiogram", "angioplasty", "stent", "catheterization",
    "coronary", "artery", "aorta", "valve", "mitral", "aortic", "tricuspid",
    "ejection", "fraction", "bypass", "pacemaker", "defibrillator",
    # Radiology
    "radiograph", "radiology", "mri", "ct", "ultrasound", "x-ray", "mammogram",
    "consolidation", "effusion", "pneumothorax", "opacity", "nodule", "mass",
    "angiography", "fluoroscopy", "pet", "scan", "contrast", "enhancement",
    # Prescriptions
    "prescription", "dosage", "tablet", "capsule", "syrup", "injection",
    "metformin", "amlodipine", "atorvastatin", "lisinopril", "omeprazole",
    "aspirin", "clopidogrel", "metoprolol", "ramipril", "amoxicillin",
    "ibuprofen", "acetaminophen", "paracetamol", "insulin", "glargine",
    "digoxin", "warfarin", "enoxaparin", "cefazolin", "azithromycin",
    # Pathology
    "biopsy", "histology", "cytology", "malignant", "benign", "metastasis",
    "carcinoma", "adenocarcinoma", "melanoma", "lymphoma", "sarcoma",
    "dysplasia", "hyperplasia", "necrosis", "inflammation", "fibrosis",
    # Surgery
    "laparoscopic", "cholecystectomy", "appendectomy", "arthroplasty",
    "hysterectomy", "mastectomy", "lobectomy", "thoracotomy", "incision",
    "suture", "anastomosis", "resection", "excision", "debridement",
    # Lab
    "hemoglobin", "hematocrit", "platelets", "leukocytes", "erythrocytes",
    "glucose", "creatinine", "bilirubin", "electrolytes", "cholesterol",
    "triglycerides", "troponin", "prothrombin", "thromboplastin",
    # General
    "diagnosis", "prognosis", "symptoms", "examination", "assessment",
    "treatment", "therapy", "prophylaxis", "contraindication", "adverse",
}

ARABIC_MEDICAL_TERMS: set = {
    # طب القلب
    "قلب", "ذبحة", "احتشاء", "نوبة", "خفقان", "ارتفاع ضغط", "سكري",
    "شريان", "تاجي", "أبهري", "تاجي", "تنظيم", "دعامة", "قسطرة",
    "صدى", "تخطيط", "كهربائي", "نبض", "عضلة", "جهاز", "بطين",
    "أذين", "صمام", "ارتجاع", "تضيق", "انسداد",
    # الأشعة
    "أشعة", "سينية", "رنين", "مغناطيسي", "مقطعي", "صدر", "صورة",
    "تليف", "انصباب", "استرواح", "كتلة", "عقدة", "صبغة",
    # الوصفات
    "وصفة", "جرعة", "قرص", "كبسولة", "معلق", "حقن", "شراب",
    "ميتفورمين", "أملوديبين", "أتورفاستاتين", "ليزينوبريل",
    "أسبرين", "كلوبيدوغريل", "أنسولين", "مضاد", "حيوي",
    # علم الأمراض
    "خزعة", "خبيث", "حميد", "ورم", "سرطان", "ميلانيني",
    "خلل", "تنسج", "التهاب", "نخر", "تليف",
    # الجراحة
    "منظار", "استئصال", "مرارة", "زائدة", "عملية", "جراحة",
    "إغلاق", "شق", "خياطة", "تخدير",
    # المختبر
    "هيموجلوبين", "كريات", "بيضاء", "حمراء", "صفائح", "جلوكوز",
    "كرياتينين", "بيليروبين", "كوليسترول", "دهون", "ثلاثية",
    # عام
    "تشخيص", "تقييم", "علاج", "فحص", "مريض", "طبيب", "مستشفى",
    "أعراض", "أدوية", "متابعة", "نتائج", "خطة",
}


@dataclass
class MetricsResult:
    """Container for all OCR quality metrics."""
    cer: float  # Character Error Rate (0.0 - 1.0+)
    wer: float  # Word Error Rate (0.0 - 1.0+)
    medical_accuracy: float  # Medical term accuracy (0.0 - 1.0)
    medical_terms_found: int = 0
    medical_terms_expected: int = 0
    correct_medical_terms: int = 0
    characters_total: int = 0
    characters_errors: int = 0
    words_total: int = 0
    words_errors: int = 0
    details: dict = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove punctuation for CER (keep alphanumeric, Arabic chars, spaces)
    text = re.sub(r"[^\w\s\u0600-\u06FF\u0750-\u077F]", "", text)
    return text


def normalize_arabic(text: str) -> str:
    """Aggressive Arabic normalization for better OCR comparison."""
    text = unicodedata.normalize("NFC", text)
    # Normalize alef variants
    text = re.sub(r"[إأآا]", "ا", text)
    # Normalize yaa
    text = re.sub(r"[ىي]", "ي", text)
    # Remove tatweel
    text = re.sub(r"\u0640", "", text)
    # Remove diacritics (tashkeel)
    diacritics = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
    text = diacritics.sub("", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_words(text: str) -> list[str]:
    """Extract words from text, handling both Arabic and English."""
    text = normalize_text(text)
    words = re.findall(r"[\u0600-\u06FF\u0750-\u077Fa-zA-Z0-9]+", text)
    return words


def character_error_rate(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """
    Calculate Character Error Rate (CER).
    CER = (S + D + I) / N
    Where S = substitutions, D = deletions, I = insertions, N = reference length.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    if not ref:
        return (0.0 if not hyp else 1.0), 0, 0

    distance = rapidfuzz.distance.Levenshtein.distance(ref, hyp)
    return distance / len(ref), distance, len(ref)


def word_error_rate(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """
    Calculate Word Error Rate (WER).
    WER = (S + D + I) / N
    Where S = substitutions, D = deletions, I = insertions, N = reference word count.
    """
    ref_words = extract_words(reference)
    hyp_words = extract_words(hypothesis)

    if not ref_words:
        return 0.0 if not hyp_words else 1.0, 0, len(hyp_words)

    distance = rapidfuzz.distance.Levenshtein.distance(ref_words, hyp_words)
    return distance / len(ref_words), distance, len(ref_words)


def medical_term_accuracy(
    reference: str,
    hypothesis: str,
    custom_terms: set | None = None,
) -> tuple[float, int, int, int]:
    """
    Calculate medical term accuracy.
    Measures how many medical terms from the reference appear in the hypothesis.
    Uses fuzzy matching with a threshold of 80% similarity.
    """
    all_terms = ENGLISH_MEDICAL_TERMS | ARABIC_MEDICAL_TERMS
    if custom_terms:
        all_terms |= custom_terms

    # Normalize both texts
    ref_normalized = normalize_text(reference) + " " + normalize_arabic(reference)
    hyp_normalized = normalize_text(hypothesis) + " " + normalize_arabic(hypothesis)

    # Extract medical terms from reference
    ref_words = set(extract_words(ref_normalized))
    found_terms = set()
    correct_terms = set()

    for word in ref_words:
        for term in all_terms:
            # Direct match
            if word == term:
                found_terms.add(term)
                break
            # Fuzzy match (80% threshold)
            similarity = rapidfuzz.fuzz.ratio(word, term)
            if similarity >= 80:
                found_terms.add(term)
                break

    # Check which found terms also appear in hypothesis
    hyp_words = set(extract_words(hyp_normalized))
    for term in found_terms:
        term_matched = False
        for word in hyp_words:
            if word == term or rapidfuzz.fuzz.ratio(word, term) >= 80:
                term_matched = True
                break
        if term_matched:
            correct_terms.add(term)

    total_expected = len(found_terms) if found_terms else 0
    total_correct = len(correct_terms)

    accuracy = total_correct / total_expected if total_expected > 0 else 0.0
    return accuracy, total_correct, total_expected, len(found_terms)


def calculate_all_metrics(
    reference: str,
    hypothesis: str,
    custom_terms: set | None = None,
) -> MetricsResult:
    """Calculate all metrics and return a MetricsResult container."""
    cer, char_errors, char_total = character_error_rate(reference, hypothesis)
    wer, word_errors, word_total = word_error_rate(reference, hypothesis)
    med_acc, med_correct, med_expected, med_found = medical_term_accuracy(
        reference, hypothesis, custom_terms
    )

    return MetricsResult(
        cer=cer,
        wer=wer,
        medical_accuracy=med_acc,
        medical_terms_found=med_found,
        medical_terms_expected=med_expected,
        correct_medical_terms=med_correct,
        characters_total=char_total,
        characters_errors=char_errors,
        words_total=word_total,
        words_errors=word_errors,
    )
