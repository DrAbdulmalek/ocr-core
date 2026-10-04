"""Benchmark medical-ocr-postprocessor correction performance.

Tests single-word correction, batch correction, phrase detection,
and overall correction quality against golden reference texts.
"""

import os
import re
import statistics

from ocr_core.benchmarks.core.metrics import EditDistance, LatencyProfiler


class CorrectionBenchmark:
    """Benchmark medical OCR postprocessor correction accuracy and speed.

    Evaluates the correction pipeline on individual words, batches of text,
    and phrase-level detection against a golden dataset.
    """

    def __init__(self, dictionary_path: str = None):
        """Initialize the correction benchmark.

        Args:
            dictionary_path: Optional path to a medical dictionary file.
                If not provided, a built-in fallback dictionary is used.
        """
        self.dictionary_path = dictionary_path
        self._postprocessor = None
        self._available: bool | None = None
        self._fallback_terms = self._build_fallback_terms()

    def _build_fallback_terms(self) -> dict[str, str]:
        """Build a minimal fallback medical term dictionary.

        Returns:
            A dict mapping common misspellings to correct terms.
        """
        return {
            "lisinoprl": "lisinopril",
            "atenolol": "atenolol",
            "metformn": "metformin",
            "atorvastatn": "atorvastatin",
            "aspirn": "aspirin",
            "adenocarcnoma": "adenocarcinoma",
            "chemothrapy": "chemotherapy",
            "myocardal": "myocardial",
            "infarcton": "infarction",
            "pneumnia": "pneumonia",
            "bronchits": "bronchitis",
            "hypertnsion": "hypertension",
            "diabetes": "diabetes",
            "hypoglycmia": "hypoglycemia",
            "arrhythmia": "arrhythmia",
            "palpitatons": "palpitations",
            "thrombosis": "thrombosis",
            "embolism": "embolism",
            "edema": "edema",
            "anemia": "anemia",
        }

    def _is_available(self) -> bool:
        """Check if the postprocessor is available.

        Returns:
            True if medical-ocr-postprocessor is importable, False otherwise.
        """
        if self._available is not None:
            return self._available

        try:
            from medical_ocr_postprocessor import MedicalOCRPostprocessor
            config = {}
            if self.dictionary_path and os.path.isfile(self.dictionary_path):
                config["dictionary_path"] = self.dictionary_path
            self._postprocessor = MedicalOCRPostprocessor(**config)
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def _correct_word(self, word: str) -> str:
        """Correct a single word using the postprocessor or fallback.

        Args:
            word: The word to correct.

        Returns:
            The corrected word.
        """
        if self._is_available() and self._postprocessor is not None:
            result = self._postprocessor.process(word)
            if isinstance(result, str):
                return result
            if isinstance(result, dict):
                return result.get("corrected", result.get("text", word))
            return str(result)
        else:
            # Fallback: use built-in dictionary
            lower = word.lower().strip()
            return self._fallback_terms.get(lower, word)

    def _correct_text(self, text: str) -> str:
        """Correct a full text string.

        Args:
            text: The text to correct.

        Returns:
            The corrected text.
        """
        if self._is_available() and self._postprocessor is not None:
            result = self._postprocessor.process(text)
            if isinstance(result, str):
                return result
            if isinstance(result, dict):
                return result.get("corrected", result.get("text", text))
            return str(result)
        else:
            # Fallback: word-by-word correction
            words = text.split()
            corrected = [self._correct_word(w) for w in words]
            return " ".join(corrected)

    def _detect_phrases(self, text: str) -> list[dict]:
        """Detect medical phrases in text.

        Uses the postprocessor if available, otherwise uses pattern matching
        against the fallback dictionary.

        Args:
            text: Input text to search for medical phrases.

        Returns:
            A list of dicts with 'phrase', 'start', 'end' keys.
        """
        if self._is_available() and self._postprocessor is not None:
            try:
                if hasattr(self._postprocessor, "detect_phrases"):
                    return self._postprocessor.detect_phrases(text)
                if hasattr(self._postprocessor, "find_medical_phrases"):
                    return self._postprocessor.find_medical_phrases(text)
            except Exception:
                pass

        # Fallback: simple regex-based phrase detection
        results = []
        text.lower()
        for term in sorted(self._fallback_terms.values(), key=len, reverse=True):
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            for match in pattern.finditer(text):
                results.append({
                    "phrase": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })

        return results

    def benchmark_single_correction(self, text: str) -> dict:
        """Benchmark single-text correction accuracy and latency.

        Args:
            text: Text to correct.

        Returns:
            A dict with corrected text, CER, and latency metrics.
        """
        profiler = LatencyProfiler(warmup_runs=3, benchmark_runs=10)
        timing = profiler.measure(self._correct_text, text)

        corrected = self._correct_text(text)
        cer_result = EditDistance.cer(text, corrected)

        return {
            "original": text,
            "corrected": corrected,
            "cer": round(cer_result["cer"], 6),
            "latency": timing,
        }

    def benchmark_batch_correction(self, texts: list[str]) -> dict:
        """Benchmark batch correction of multiple texts.

        Args:
            texts: List of texts to correct.

        Returns:
            A dict with per-text CER, mean CER, and timing metrics.
        """
        if not texts:
            return {"error": "Empty text list", "mean_cer": None}

        profiler = LatencyProfiler(warmup_runs=1, benchmark_runs=5)

        def batch_correct():
            return [self._correct_text(t) for t in texts]

        timing = profiler.measure(batch_correct)
        corrected_texts = batch_correct()

        cer_scores = []
        for original, corrected in zip(texts, corrected_texts):
            cer_result = EditDistance.cer(original, corrected)
            cer_scores.append(cer_result["cer"])

        return {
            "total_texts": len(texts),
            "mean_cer": round(statistics.mean(cer_scores), 6) if cer_scores else None,
            "min_cer": round(min(cer_scores), 6) if cer_scores else None,
            "max_cer": round(max(cer_scores), 6) if cer_scores else None,
            "cer_scores": [round(c, 6) for c in cer_scores],
            "latency": timing,
        }

    def benchmark_phrase_detection(self, texts: list[str]) -> dict:
        """Benchmark medical phrase detection accuracy and speed.

        Args:
            texts: List of texts to scan for medical phrases.

        Returns:
            A dict with detection counts, latencies, and coverage metrics.
        """
        if not texts:
            return {"error": "Empty text list"}

        profiler = LatencyProfiler(warmup_runs=1, benchmark_runs=5)

        def detect_all():
            return [self._detect_phrases(t) for t in texts]

        timing = profiler.measure(detect_all)
        all_phrases = detect_all()

        phrase_counts = [len(p) for p in all_phrases]

        return {
            "total_texts": len(texts),
            "total_phrases_found": sum(phrase_counts),
            "mean_phrases_per_text": round(statistics.mean(phrase_counts), 2) if phrase_counts else 0,
            "max_phrases": max(phrase_counts) if phrase_counts else 0,
            "min_phrases": min(phrase_counts) if phrase_counts else 0,
            "latency": timing,
        }

    def run(self, golden_dataset: dict) -> dict:
        """Run correction benchmark against a golden dataset.

        Processes each test case through the correction pipeline and
        measures accuracy against reference texts.

        Args:
            golden_dataset: A golden dataset dict with 'test_cases' key.

        Returns:
            A dict with aggregate correction benchmark metrics.
        """
        test_cases = golden_dataset.get("test_cases", [])
        if not test_cases:
            return {
                "engine": "correction",
                "status": "error",
                "error": "No test cases in golden dataset",
            }

        texts = [case.get("hypothesis", "") for case in test_cases]
        references = [case.get("reference", "") for case in test_cases]

        # Batch correction
        batch_result = self.benchmark_batch_correction(texts)

        # Per-case correction against reference
        cer_scores = []
        for ref, hyp in zip(references, texts):
            corrected = self._correct_text(hyp)
            cer_result = EditDistance.cer(ref, corrected)
            cer_scores.append(cer_result["cer"])

        # Phrase detection
        phrase_result = self.benchmark_phrase_detection(texts)

        return {
            "engine": "correction",
            "status": "success",
            "total_cases": len(test_cases),
            "postprocessor_available": self._is_available(),
            "mean_correction_cer": round(statistics.mean(cer_scores), 6) if cer_scores else None,
            "min_correction_cer": round(min(cer_scores), 6) if cer_scores else None,
            "max_correction_cer": round(max(cer_scores), 6) if cer_scores else None,
            "per_case_cer": [round(c, 6) for c in cer_scores],
            "batch_mean_cer": batch_result.get("mean_cer"),
            "phrases_detected": phrase_result.get("total_phrases_found", 0),
            "mean_phrases_per_text": phrase_result.get("mean_phrases_per_text", 0),
            "mean_latency_s": batch_result.get("latency", {}).get("mean"),
        }
