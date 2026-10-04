"""Core benchmark components: metrics, runner, and reporter."""

from ocr_core.benchmarks.core.metrics import (
    BenchmarkSuite,
    EditDistance,
    LatencyProfiler,
    MedicalTermEvaluator,
)
from ocr_core.benchmarks.core.reporter import BenchmarkReporter
from ocr_core.benchmarks.core.runner import BenchmarkRunner

__all__ = [
    "EditDistance",
    "LatencyProfiler",
    "MedicalTermEvaluator",
    "BenchmarkSuite",
    "BenchmarkRunner",
    "BenchmarkReporter",
]
