"""EasyOCR Benchmark.

Benchmarks EasyOCR engine on medical documents, measuring
CER, WER, medical term accuracy, and processing latency.
"""

import statistics

from ocr_core.benchmarks.core.metrics import EditDistance, LatencyProfiler, MedicalTermEvaluator


class EasyOCRBenchmark:
    """Benchmark EasyOCR on medical text/images.

    Can run against text pairs from a golden dataset or perform
    actual image-based OCR if images are available.
    """

    def __init__(self, languages: list[str] = None, gpu: bool = False):
        """Initialize EasyOCR benchmark.

        Args:
            languages: List of language codes for EasyOCR (e.g. ['en', 'ar']).
            gpu: Whether to use GPU acceleration.
        """
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader = None
        self._available: bool | None = None

    def _is_available(self) -> bool:
        """Check if EasyOCR is installed.

        Returns:
            True if easyocr can be imported, False otherwise.
        """
        if self._available is not None:
            return self._available

        try:
            import easyocr  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def _get_reader(self):
        """Lazy-load the EasyOCR reader.

        Returns:
            An EasyOCR Reader instance.
        """
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def benchmark_text(self, reference: str, hypothesis: str) -> dict:
        """Benchmark a single text pair for CER and WER.

        Args:
            reference: Ground truth text.
            hypothesis: OCR output text.

        Returns:
            A dict with CER, WER, and related metrics.
        """
        cer_result = EditDistance.cer(reference, hypothesis)
        wer_result = EditDistance.wer(reference, hypothesis)

        return {
            "cer": round(cer_result["cer"], 6),
            "wer": round(wer_result["wer"], 6),
            "cer_substitutions": cer_result["substitutions"],
            "cer_deletions": cer_result["deletions"],
            "cer_insertions": cer_result["insertions"],
            "wer_substitutions": wer_result["substitutions"],
            "wer_deletions": wer_result["deletions"],
            "wer_insertions": wer_result["insertions"],
        }

    def benchmark_image(self, image_path: str) -> dict:
        """Run EasyOCR on an actual image.

        Args:
            image_path: Path to the image to process.

        Returns:
            A dict with extracted text, confidence, and latency.
        """
        reader = self._get_reader()
        profiler = LatencyProfiler(warmup_runs=1, benchmark_runs=3)

        def run_ocr():
            results = reader.readtext(image_path)
            return results

        timing = profiler.measure(run_ocr)
        ocr_results = run_ocr()

        texts = [item[1] for item in ocr_results]
        confidences = [item[2] for item in ocr_results]
        full_text = " ".join(texts)

        return {
            "text": full_text,
            "num_blocks": len(texts),
            "avg_confidence": round(statistics.mean(confidences), 4) if confidences else 0.0,
            "min_confidence": round(min(confidences), 4) if confidences else 0.0,
            "max_confidence": round(max(confidences), 4) if confidences else 0.0,
            "latency": timing,
        }

    def run(self, golden_dataset: dict) -> dict:
        """Run EasyOCR benchmark against a golden dataset.

        Evaluates each test case in the dataset, computing aggregate
        CER, WER, medical term accuracy, and latency.

        Args:
            golden_dataset: A golden dataset dict with 'test_cases' key.

        Returns:
            A dict with aggregate benchmark metrics.
        """
        if not self._is_available():
            return {
                "engine": "easyocr",
                "status": "unavailable",
                "error": "EasyOCR not installed. Install with: pip install easyocr",
            }

        test_cases = golden_dataset.get("test_cases", [])
        if not test_cases:
            return {
                "engine": "easyocr",
                "status": "error",
                "error": "No test cases in golden dataset",
            }

        cer_scores = []
        wer_scores = []
        term_accuracies = []
        latencies = []

        for case in test_cases:
            reference = case.get("reference", "")
            hypothesis = case.get("hypothesis", "")
            medical_terms = case.get("medical_terms", [])

            cer_result = EditDistance.cer(reference, hypothesis)
            wer_result = EditDistance.wer(reference, hypothesis)
            cer_scores.append(cer_result["cer"])
            wer_scores.append(wer_result["wer"])

            if medical_terms:
                evaluator = MedicalTermEvaluator(medical_terms)
                term_eval = evaluator.evaluate(reference, hypothesis)
                term_accuracies.append(term_eval["accuracy"])

            profiler = LatencyProfiler(warmup_runs=1, benchmark_runs=5)
            latency = profiler.measure(
                lambda r=reference, h=hypothesis: (
                    EditDistance.cer(r, h),
                    EditDistance.wer(r, h),
                )
            )
            latencies.append(latency["mean"])

        return {
            "engine": "easyocr",
            "status": "success",
            "total_cases": len(test_cases),
            "mean_cer": round(statistics.mean(cer_scores), 6) if cer_scores else None,
            "mean_wer": round(statistics.mean(wer_scores), 6) if wer_scores else None,
            "mean_medical_term_accuracy": round(statistics.mean(term_accuracies), 6) if term_accuracies else None,
            "mean_latency_s": round(statistics.mean(latencies), 6) if latencies else None,
            "cer_scores": [round(c, 6) for c in cer_scores],
            "wer_scores": [round(w, 6) for w in wer_scores],
        }
