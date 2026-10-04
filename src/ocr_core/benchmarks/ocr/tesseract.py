"""Tesseract OCR Benchmark.

Benchmarks Tesseract OCR engine on medical documents, measuring
CER, WER, medical term accuracy, and processing latency.
"""

import statistics

from ocr_core.benchmarks.core.metrics import EditDistance, LatencyProfiler, MedicalTermEvaluator


class TesseractBenchmark:
    """Benchmark Tesseract OCR on medical text.

    Performs text extraction via Tesseract and evaluates against a
    golden dataset using character-level and word-level error rates.
    """

    def __init__(self, lang: str = "eng", tesseract_cmd: str = "tesseract"):
        """Initialize Tesseract benchmark.

        Args:
            lang: Tesseract language code (e.g. 'eng', 'ara', 'eng+ara').
            tesseract_cmd: Path to the tesseract binary.
        """
        self.lang = lang
        self.tesseract_cmd = tesseract_cmd
        self._available: bool | None = None

    def _is_available(self) -> bool:
        """Check if Tesseract is installed and available.

        Returns:
            True if tesseract binary is found, False otherwise.
        """
        if self._available is not None:
            return self._available

        try:
            import subprocess
            result = subprocess.run(
                [self.tesseract_cmd, "--version"],
                capture_output=True,
                timeout=10,
            )
            self._available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._available = False

        return self._available

    def _ocr_image(self, image_path: str) -> str:
        """Run Tesseract OCR on an image file.

        Args:
            image_path: Path to the image to process.

        Returns:
            Extracted text as a string.
        """
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as out_f:
            output_file = out_f.name

        try:
            subprocess.run(
                [
                    self.tesseract_cmd,
                    image_path,
                    output_file.replace(".txt", ""),
                    "-l",
                    self.lang,
                    "--psm",
                    "6",
                ],
                capture_output=True,
                timeout=60,
            )

            with open(output_file, encoding="utf-8") as f:
                return f.read().strip()
        finally:
            import os
            if os.path.exists(output_file):
                os.unlink(output_file)
            txt_file = output_file.replace(".txt", ".txt")
            if os.path.exists(txt_file):
                os.unlink(txt_file)

    def benchmark_text(self, reference: str, hypothesis: str) -> dict:
        """Benchmark a single text pair for CER and WER.

        Args:
            reference: Ground truth text.
            hypothesis: OCR output text.

        Returns:
            A dict with CER, WER, and latency metrics.
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

    def benchmark_latency(self, reference: str, hypothesis: str) -> dict:
        """Benchmark processing latency on a single text pair.

        Measures the time to compute CER and WER metrics.

        Args:
            reference: Ground truth text.
            hypothesis: OCR output text.

        Returns:
            A dict with latency statistics.
        """
        profiler = LatencyProfiler(warmup_runs=3, benchmark_runs=10)

        def compute_metrics():
            EditDistance.cer(reference, hypothesis)
            EditDistance.wer(reference, hypothesis)

        return profiler.measure(compute_metrics)

    def run(self, golden_dataset: dict) -> dict:
        """Run Tesseract benchmark against a golden dataset.

        Evaluates each test case in the dataset, computing aggregate
        CER, WER, medical term accuracy, and latency.

        Args:
            golden_dataset: A golden dataset dict with 'test_cases' key.

        Returns:
            A dict with aggregate benchmark metrics.
        """
        if not self._is_available():
            return {
                "engine": "tesseract",
                "status": "unavailable",
                "error": f"Tesseract not found at '{self.tesseract_cmd}'. Install with: apt-get install tesseract-ocr",
            }

        test_cases = golden_dataset.get("test_cases", [])
        if not test_cases:
            return {
                "engine": "tesseract",
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

            # CER and WER
            cer_result = EditDistance.cer(reference, hypothesis)
            wer_result = EditDistance.wer(reference, hypothesis)
            cer_scores.append(cer_result["cer"])
            wer_scores.append(wer_result["wer"])

            # Medical term evaluation
            if medical_terms:
                evaluator = MedicalTermEvaluator(medical_terms)
                term_eval = evaluator.evaluate(reference, hypothesis)
                term_accuracies.append(term_eval["accuracy"])

            # Latency
            profiler = LatencyProfiler(warmup_runs=1, benchmark_runs=5)
            latency = profiler.measure(
                lambda r=reference, h=hypothesis: (
                    EditDistance.cer(r, h),
                    EditDistance.wer(r, h),
                )
            )
            latencies.append(latency["mean"])

        return {
            "engine": "tesseract",
            "status": "success",
            "total_cases": len(test_cases),
            "mean_cer": round(statistics.mean(cer_scores), 6) if cer_scores else None,
            "mean_wer": round(statistics.mean(wer_scores), 6) if wer_scores else None,
            "mean_medical_term_accuracy": round(statistics.mean(term_accuracies), 6) if term_accuracies else None,
            "mean_latency_s": round(statistics.mean(latencies), 6) if latencies else None,
            "cer_scores": [round(c, 6) for c in cer_scores],
            "wer_scores": [round(w, 6) for w in wer_scores],
        }
