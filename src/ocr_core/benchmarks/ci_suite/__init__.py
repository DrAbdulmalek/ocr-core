"""
medical-ocr-benchmarks — Benchmark suite for measuring OCR quality on medical documents.
Benchmark Suite لقياس جودة التعرف البصري على الوثائق الطبية.

This package provides tools to:
- Run OCR benchmarks across multiple engines (PaddleOCR, Tesseract, EasyOCR, TrOCR, Surya)
- Calculate metrics: CER, WER, medical term accuracy, latency, throughput
- Generate reports in Markdown, JSON, and HTML
- Check CI thresholds for automatic failure detection
- Manage test datasets across specialties and languages
"""

__version__ = "0.1.0"
__author__ = "Dr. Abdulmalek"

from ocr_core.benchmarks.ci_suite.ci import ThresholdChecker
from ocr_core.benchmarks.ci_suite.dataset import DatasetManager
from ocr_core.benchmarks.ci_suite.metrics import (
    calculate_all_metrics,
    character_error_rate,
    medical_term_accuracy,
    word_error_rate,
)
from ocr_core.benchmarks.ci_suite.report import ReportGenerator
from ocr_core.benchmarks.ci_suite.runner import BenchmarkRunner

__all__ = [
    "character_error_rate",
    "word_error_rate",
    "medical_term_accuracy",
    "calculate_all_metrics",
    "DatasetManager",
    "BenchmarkRunner",
    "ReportGenerator",
    "ThresholdChecker",
]
