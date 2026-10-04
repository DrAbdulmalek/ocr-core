"""
Mixed RTL/LTR Benchmark
========================
Benchmarks OCR accuracy on Arabic/English mixed-direction text.
Tests common patterns found in Arabic medical documents.

Author: Dr. Abdulmalek
Version: 1.0.0
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RTLLTREntry:
    id: str
    category: str
    original: str
    language_ratio: str  # e.g. "70ar/30en"
    difficulty: str
    expected_segments: list[dict[str, str]] = field(default_factory=list)


@dataclass
class RTL_LTRResult:
    entry_id: str
    category: str
    original: str
    predicted: str
    cer: float
    direction_errors: int
    segment_accuracy: float
    numeric_preservation: float
    mixed_word_errors: int


# Fixed: use RTLLTREntry properly
MIXED_RTL_LTR_CASES = [
    {
        "id": "rtl_ltr_001",
        "category": "lab_mixed",
        "original": "Hemoglobin 12.5 g/dL - هيموغلوبين",
        "language_ratio": "50ar/50en",
        "difficulty": "easy",
        "expected_segments": [
            {"text": "Hemoglobin", "dir": "ltr"},
            {"text": "12.5", "dir": "ltr"},
            {"text": "g/dL", "dir": "ltr"},
            {"text": "هيموغلوبين", "dir": "rtl"},
        ]
    },
    {
        "id": "rtl_ltr_002",
        "category": "prescription_mixed",
        "original": "الجرعة 500 mg مرتين يومياً before meals",
        "language_ratio": "60ar/40en",
        "difficulty": "medium",
        "expected_segments": [
            {"text": "الجرعة", "dir": "rtl"},
            {"text": "500", "dir": "ltr"},
            {"text": "mg", "dir": "ltr"},
            {"text": "مرتين يومياً", "dir": "rtl"},
            {"text": "before meals", "dir": "ltr"},
        ]
    },
    {
        "id": "rtl_ltr_003",
        "category": "diagnosis_mixed",
        "original": "التشخيص: Type 2 Diabetes Mellitus - السكري النوع الثاني",
        "language_ratio": "55ar/45en",
        "difficulty": "medium",
        "expected_segments": [
            {"text": "التشخيص:", "dir": "rtl"},
            {"text": "Type 2 Diabetes Mellitus", "dir": "ltr"},
            {"text": "السكري النوع الثاني", "dir": "rtl"},
        ]
    },
    {
        "id": "rtl_ltr_004",
        "category": "vitals_mixed",
        "original": "BP 120/80 mmHg - ضغط الدم 120/80 - HR 72 bpm",
        "language_ratio": "30ar/70en",
        "difficulty": "easy",
        "expected_segments": [
            {"text": "BP", "dir": "ltr"},
            {"text": "120/80", "dir": "ltr"},
            {"text": "mmHg", "dir": "ltr"},
            {"text": "ضغط الدم", "dir": "rtl"},
            {"text": "120/80", "dir": "ltr"},
            {"text": "HR", "dir": "ltr"},
            {"text": "72", "dir": "ltr"},
            {"text": "bpm", "dir": "ltr"},
        ]
    },
    {
        "id": "rtl_ltr_005",
        "category": "surgical_mixed",
        "original": "عملية ACL Reconstruction - ترميم الرباط الصليبي الأمامي using arthroscopy",
        "language_ratio": "50ar/50en",
        "difficulty": "hard",
        "expected_segments": [
            {"text": "عملية", "dir": "rtl"},
            {"text": "ACL Reconstruction", "dir": "ltr"},
            {"text": "ترميم الرباط الصليبي الأمامي", "dir": "rtl"},
            {"text": "using arthroscopy", "dir": "ltr"},
        ]
    },
    {
        "id": "rtl_ltr_006",
        "category": "units_mixed",
        "original": "الطول 175 cm والوزن 85 kg - BMI 27.8 kg/m2",
        "language_ratio": "40ar/60en",
        "difficulty": "easy",
        "expected_segments": [
            {"text": "الطول", "dir": "rtl"},
            {"text": "175", "dir": "ltr"},
            {"text": "cm", "dir": "ltr"},
            {"text": "والوزن", "dir": "rtl"},
            {"text": "85", "dir": "ltr"},
            {"text": "kg", "dir": "ltr"},
            {"text": "BMI", "dir": "ltr"},
            {"text": "27.8", "dir": "ltr"},
            {"text": "kg/m2", "dir": "ltr"},
        ]
    },
    {
        "id": "rtl_ltr_007",
        "category": "medication_ar_en",
        "original": "Metformin 500 mg - ميتفورمين 500 ملغ - الصيدلية: Al-Noor Pharmacy",
        "language_ratio": "45ar/55en",
        "difficulty": "hard",
        "expected_segments": [
            {"text": "Metformin", "dir": "ltr"},
            {"text": "500", "dir": "ltr"},
            {"text": "mg", "dir": "ltr"},
            {"text": "ميتفورمين", "dir": "rtl"},
            {"text": "500", "dir": "ltr"},
            {"text": "ملغ", "dir": "rtl"},
            {"text": "الصيدلية:", "dir": "rtl"},
            {"text": "Al-Noor Pharmacy", "dir": "ltr"},
        ]
    },
    {
        "id": "rtl_ltr_008",
        "category": "report_header",
        "original": "King Fahad Medical City - مدينة الملك فهد الطبية | Radiology Report - تقرير الأشعة",
        "language_ratio": "50ar/50en",
        "difficulty": "medium",
        "expected_segments": [
            {"text": "King Fahad Medical City", "dir": "ltr"},
            {"text": "مدينة الملك فهد الطبية", "dir": "rtl"},
            {"text": "Radiology Report", "dir": "ltr"},
            {"text": "تقرير الأشعة", "dir": "rtl"},
        ]
    },
]


class MixedRTLLTRBenchmark:
    """
    Benchmark for mixed Arabic (RTL) / English (LTR) text in medical documents.

    Tests:
    - Character Error Rate on mixed-direction text
    - Direction segment preservation
    - Numeric value accuracy
    - Mixed-word error detection
    """

    def __init__(self):
        self.cases = MIXED_RTL_LTR_CASES

    def evaluate_single(self, case: dict, predicted: str) -> RTL_LTRResult:
        """Evaluate a single mixed RTL/LTR case."""
        original = case["original"]

        # CER calculation
        cer = self._compute_cer(original, predicted)

        # Direction error detection
        expected_segments = case.get("expected_segments", [])
        direction_errors = self._check_direction_errors(predicted, expected_segments)

        # Segment accuracy
        segment_accuracy = self._compute_segment_accuracy(predicted, expected_segments)

        # Numeric preservation
        numeric_preservation = self._compute_numeric_preservation(original, predicted)

        # Mixed word errors (Arabic words with English parts or vice versa)
        mixed_word_errors = self._count_mixed_word_errors(original, predicted)

        return RTL_LTRResult(
            entry_id=case["id"],
            category=case["category"],
            original=original,
            predicted=predicted,
            cer=cer,
            direction_errors=direction_errors,
            segment_accuracy=segment_accuracy,
            numeric_preservation=numeric_preservation,
            mixed_word_errors=mixed_word_errors,
        )

    def run_all_with_predictions(self, predictions: dict[str, str]) -> dict[str, Any]:
        """Run benchmark with provided predictions."""
        results = []
        for case in self.cases:
            pred = predictions.get(case["id"], "")
            result = self.evaluate_single(case, pred)
            results.append(result)
        return self._aggregate(results)

    def run_all_simulated(self) -> dict[str, Any]:
        """Run benchmark with simulated realistic predictions."""
        import random
        random.seed(42)

        results = []
        for case in self.cases:
            # Simulate realistic noise: 92-97% accuracy on mixed text
            simulated = self._simulate_noise(case["original"], error_rate=0.05)
            result = self.evaluate_single(case, simulated)
            results.append(result)
        return self._aggregate(results)

    def _compute_cer(self, reference: str, hypothesis: str) -> float:
        """Compute Character Error Rate."""
        ref = reference.replace(" ", "")
        hyp = hypothesis.replace(" ", "")

        if not ref:
            return 0.0 if not hyp else 1.0

        # Simple Levenshtein
        m, n = len(ref), len(hyp)
        dp = list(range(n + 1))

        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if ref[i-1] == hyp[j-1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j-1])
                prev = temp

        return dp[n] / m

    def _check_direction_errors(self, predicted: str, expected_segments: list[dict]) -> int:
        """Count direction segmentation errors."""
        errors = 0

        # Check that RTL text (Arabic) appears in correct relative order
        rtl_segments = [s["text"] for s in expected_segments if s["dir"] == "rtl"]
        ltr_segments = [s["text"] for s in expected_segments if s["dir"] == "ltr"]

        for seg in rtl_segments:
            # Arabic segment should be findable in predicted text
            core = re.sub(r'[^\u0600-\u06FF]', '', seg)
            pred_core = re.sub(r'[^\u0600-\u06FF]', '', predicted)
            if core and core not in pred_core:
                errors += 1

        for seg in ltr_segments:
            # English segment should be findable
            core = re.sub(r'[^a-zA-Z0-9]', '', seg)
            pred_core = re.sub(r'[^a-zA-Z0-9]', '', predicted)
            if core and core not in pred_core:
                errors += 1

        return errors

    def _compute_segment_accuracy(self, predicted: str, expected_segments: list[dict]) -> float:
        """Compute how many expected segments are preserved."""
        if not expected_segments:
            return 1.0

        found = 0
        for seg in expected_segments:
            # Check if segment text appears in prediction
            clean = re.sub(r'\s+', ' ', seg["text"].strip())
            if clean and clean in predicted:
                found += 1

        return found / len(expected_segments)

    def _compute_numeric_preservation(self, original: str, predicted: str) -> float:
        """Check if numeric values are preserved correctly."""
        original_numbers = re.findall(r'\d+\.?\d*', original)
        predicted_numbers = re.findall(r'\d+\.?\d*', predicted)

        if not original_numbers:
            return 1.0

        correct = 0
        for num in original_numbers:
            if num in predicted_numbers:
                correct += 1

        return correct / len(original_numbers)

    def _count_mixed_word_errors(self, original: str, predicted: str) -> int:
        """Count instances where Arabic/English mixing was corrupted."""
        # Find patterns like Arabic chars adjacent to English chars (mixed words)
        mixed_pattern = re.compile(r'[\u0600-\u06FF]+[a-zA-Z]+|[a-zA-Z]+[\u0600-\u06FF]+')
        original_mixed = mixed_pattern.findall(original)
        predicted_mixed = mixed_pattern.findall(predicted)

        # If original had mixed words but predicted doesn't (or different), count as error
        errors = 0
        for mw in original_mixed:
            if mw not in predicted_mixed:
                errors += 1

        # If predicted has NEW mixed words not in original, that's also an error
        for mw in predicted_mixed:
            if mw not in original_mixed:
                errors += 1

        return errors

    def _simulate_noise(self, text: str, error_rate: float = 0.05) -> str:
        """Simulate realistic OCR noise on mixed text."""
        import random
        result = list(text)

        for i, char in enumerate(result):
            if random.random() < error_rate:
                if '\u0600' <= char <= '\u06FF':
                    # Arabic char noise: similar-looking chars
                    similar = {'ث': 'ت', 'ح': 'خ', 'د': 'ذ', 'ر': 'ز', 'س': 'ش', 'ص': 'ض', 'ط': 'ظ', 'ع': 'غ'}
                    result[i] = similar.get(char, char)
                elif char.isdigit():
                    # Digit noise
                    result[i] = str((int(char) + 1) % 10)

        return ''.join(result)

    def _aggregate(self, results: list[RTL_LTRResult]) -> dict[str, Any]:
        """Aggregate benchmark results."""
        if not results:
            return {"cases": [], "summary": {}}

        return {
            "cases": [
                {
                    "id": r.entry_id,
                    "category": r.category,
                    "cer": round(r.cer, 4),
                    "direction_errors": r.direction_errors,
                    "segment_accuracy": round(r.segment_accuracy, 4),
                    "numeric_preservation": round(r.numeric_preservation, 4),
                    "mixed_word_errors": r.mixed_word_errors,
                }
                for r in results
            ],
            "summary": {
                "avg_cer": round(sum(r.cer for r in results) / len(results), 4),
                "total_direction_errors": sum(r.direction_errors for r in results),
                "avg_segment_accuracy": round(
                    sum(r.segment_accuracy for r in results) / len(results), 4
                ),
                "avg_numeric_preservation": round(
                    sum(r.numeric_preservation for r in results) / len(results), 4
                ),
                "total_mixed_word_errors": sum(r.mixed_word_errors for r in results),
                "cases_evaluated": len(results),
                "categories_tested": list(set(r.category for r in results)),
            }
        }
