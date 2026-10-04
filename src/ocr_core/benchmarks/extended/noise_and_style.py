"""
Benchmarks for noise types, expanded specialties, and handwriting styles.
These benchmarks test OCR robustness under real-world medical document conditions.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtendedBenchmarkResult:
    """Result from an extended benchmark case."""
    case_id: str
    category: str  # noise_type, specialty, or handwriting_style
    ground_truth: str
    hypothesis: str = ""
    cer: float = -1.0
    passed: bool = False
    notes: str = ""


class NoiseTypeBenchmark:
    """Benchmark OCR accuracy under specific noise conditions."""

    CATEGORIES = [
        "motion_blur", "out_of_focus", "rotation_skew", "low_resolution",
        "watermark_overlay", "fax_artifacts", "stamp_annotation", "grid_lines"
    ]

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir or os.path.join(
            os.path.dirname(__file__), "..", "data", "golden"
        ))
        self.cases: list[dict] = []
        self._load()

    def _load(self):
        path = self.data_dir / "noise_types.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.cases = data.get("cases", [])

    def run(self, ocr_func) -> list[ExtendedBenchmarkResult]:
        """
        Run benchmark with an OCR function.

        Args:
            ocr_func: callable(text: str) -> str that applies noise simulation + OCR
                     For text-only testing, this can be a function that introduces
                     noise patterns and then attempts to correct them.

        Returns:
            List of benchmark results per case.
        """
        results = []
        for case in self.cases:
            hypothesis = ocr_func(case["ground_truth"])
            cer = self._compute_cer(case["ground_truth"], hypothesis)
            results.append(ExtendedBenchmarkResult(
                case_id=case["id"],
                category="noise_type",
                ground_truth=case["ground_truth"],
                hypothesis=hypothesis,
                cer=cer,
                passed=cer < 0.20,
                notes=f'noise_type={case["noise_type"]}, difficulty={case["difficulty"]}'
            ))
        return results

    def get_noise_type_summary(self, results: list[ExtendedBenchmarkResult]) -> dict:
        """Aggregate results by noise type."""
        summary = {}
        for r in results:
            noise_type = r.notes.split("=")[1].split(",")[0] if "=" in r.notes else "unknown"
            if noise_type not in summary:
                summary[noise_type] = {"cases": 0, "avg_cer": 0.0, "passed": 0}
            summary[noise_type]["cases"] += 1
            summary[noise_type]["avg_cer"] += r.cer
            if r.passed:
                summary[noise_type]["passed"] += 1
        for k in summary:
            s = summary[k]
            s["avg_cer"] = round(s["avg_cer"] / max(s["cases"], 1), 4)
            s["pass_rate"] = round(s["passed"] / max(s["cases"], 1), 2)
        return summary

    @staticmethod
    def _compute_cer(reference: str, hypothesis: str) -> float:
        """Compute Character Error Rate using Levenshtein distance."""
        if not reference:
            return 0.0 if not hypothesis else 1.0
        m, n = len(reference), len(hypothesis)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if reference[i-1] == hypothesis[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
        return dp[m][n] / m


class HandwritingStyleBenchmark:
    """Benchmark OCR accuracy across handwriting styles."""

    STYLES = [
        "arabic_print", "arabic_cursive", "english_print", "english_cursive",
        "doctor_scratch", "mixed_script", "mixed_numerals",
        "medical_shorthand", "marginal_notes"
    ]

    DIFFICULTY_MAP = {"easy": 0.10, "medium": 0.15, "hard": 0.25, "very_hard": 0.35}

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir or os.path.join(
            os.path.dirname(__file__), "..", "data", "golden"
        ))
        self.cases: list[dict] = []
        self._load()

    def _load(self):
        path = self.data_dir / "handwriting_styles.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.cases = data.get("cases", [])

    def run(self, ocr_func) -> list[ExtendedBenchmarkResult]:
        """Run benchmark. ocr_func should simulate or perform OCR on text."""
        results = []
        for case in self.cases:
            hypothesis = ocr_func(case["ground_truth"])
            cer = NoiseTypeBenchmark._compute_cer(case["ground_truth"], hypothesis)
            threshold = self.DIFFICULTY_MAP.get(case.get("difficulty", "medium"), 0.15)
            results.append(ExtendedBenchmarkResult(
                case_id=case["id"],
                category="handwriting_style",
                ground_truth=case["ground_truth"],
                hypothesis=hypothesis,
                cer=cer,
                passed=cer < threshold,
                notes=f'style={case["style"]}, difficulty={case["difficulty"]}, threshold={threshold}'
            ))
        return results


class SpecialtyBenchmark:
    """Benchmark OCR accuracy for expanded medical specialties."""

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir or os.path.join(
            os.path.dirname(__file__), "..", "data", "golden"
        ))
        self.cases: list[dict] = []
        self._load()

    def _load(self):
        path = self.data_dir / "specialties_expanded.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.cases = data.get("cases", [])

    def run(self, ocr_func) -> list[ExtendedBenchmarkResult]:
        """Run benchmark per specialty."""
        results = []
        for case in self.cases:
            hypothesis = ocr_func(case["ground_truth"])
            cer = NoiseTypeBenchmark._compute_cer(case["ground_truth"], hypothesis)
            results.append(ExtendedBenchmarkResult(
                case_id=case["id"],
                category="specialty",
                ground_truth=case["ground_truth"],
                hypothesis=hypothesis,
                cer=cer,
                passed=cer < 0.18,
                notes=f'specialty={case["specialty"]}, language={case["language"]}'
            ))
        return results

    def get_specialty_summary(self, results: list[ExtendedBenchmarkResult]) -> dict:
        """Aggregate results by specialty."""
        summary = {}
        for r in results:
            spec = r.notes.split("=")[1].split(",")[0] if "=" in r.notes else "unknown"
            if spec not in summary:
                summary[spec] = {"cases": 0, "avg_cer": 0.0, "passed": 0}
            summary[spec]["cases"] += 1
            summary[spec]["avg_cer"] += r.cer
            if r.passed:
                summary[spec]["passed"] += 1
        for k in summary:
            s = summary[k]
            s["avg_cer"] = round(s["avg_cer"] / max(s["cases"], 1), 4)
            s["pass_rate"] = round(s["passed"] / max(s["cases"], 1), 2)
        return summary
