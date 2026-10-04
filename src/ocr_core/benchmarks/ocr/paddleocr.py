"""PaddleOCR Benchmark.

Benchmarks PaddleOCR engine on medical documents, measuring
CER, WER, medical term accuracy, and processing latency.
"""

import statistics

from ocr_core.benchmarks.core.metrics import EditDistance, LatencyProfiler, MedicalTermEvaluator


class PaddleOCRBenchmark:
    """Benchmark PaddleOCR on medical text/images.

    Supports both text-pair evaluation from golden datasets and
    actual image-based OCR when the PaddleOCR library is available.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        """Initialize PaddleOCR benchmark.

        Args:
            lang: Language for PaddleOCR ('en', 'ar', 'ch', etc.).
            use_gpu: Whether to use GPU acceleration.
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self._ocr = None
        self._available: bool | None = None

    def _is_available(self) -> bool:
        """Check if PaddleOCR is installed.

        Returns:
            True if paddleocr can be imported, False otherwise.
        """
        if self._available is not None:
            return self._available

        try:
            from paddleocr import PaddleOCR  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def _get_ocr(self):
        """Lazy-load the PaddleOCR engine.

        Returns:
            A PaddleOCR instance.
        """
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
            )
        return self._ocr

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
        """Run PaddleOCR on an actual image.

        Args:
            image_path: Path to the image to process.

        Returns:
            A dict with extracted text, confidence, and latency.
        """
        ocr = self._get_ocr()
        profiler = LatencyProfiler(warmup_runs=1, benchmark_runs=3)

        def run_ocr():
            result = ocr.ocr(image_path, cls=True)
            return result

        timing = profiler.measure(run_ocr)
        ocr_results = run_ocr()

        texts = []
        confidences = []
        if ocr_results and ocr_results[0]:
            for line in ocr_results[0]:
                if len(line) >= 2:
                    texts.append(line[1][0])
                    confidences.append(line[1][1])

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
        """Run PaddleOCR benchmark against a golden dataset.

        Evaluates each test case in the dataset, computing aggregate
        CER, WER, medical term accuracy, and latency.

        Args:
            golden_dataset: A golden dataset dict with 'test_cases' key.

        Returns:
            A dict with aggregate benchmark metrics.
        """
        if not self._is_available():
            return {
                "engine": "paddleocr",
                "status": "unavailable",
                "error": "PaddleOCR not installed. Install with: pip install paddleocr",
            }

        test_cases = golden_dataset.get("test_cases", [])
        if not test_cases:
            return {
                "engine": "paddleocr",
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
            "engine": "paddleocr",
            "status": "success",
            "total_cases": len(test_cases),
            "mean_cer": round(statistics.mean(cer_scores), 6) if cer_scores else None,
            "mean_wer": round(statistics.mean(wer_scores), 6) if wer_scores else None,
            "mean_medical_term_accuracy": round(statistics.mean(term_accuracies), 6) if term_accuracies else None,
            "mean_latency_s": round(statistics.mean(latencies), 6) if latencies else None,
            "cer_scores": [round(c, 6) for c in cer_scores],
            "wer_scores": [round(w, 6) for w in wer_scores],
        }
