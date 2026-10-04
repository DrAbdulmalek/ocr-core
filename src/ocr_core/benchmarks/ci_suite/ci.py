"""
ci.py — CI threshold checking for automatic failure detection.
فحص أطراف CI للكشف التلقائي عن الفشل.

Compares benchmark results against configured thresholds and baseline snapshots.
Fails if CER/WER degrades beyond acceptable limits.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ocr_core.benchmarks.ci_suite.report import EngineResult


@dataclass
class ThresholdViolation:
    """A single threshold violation."""
    engine: str
    metric: str
    value: float
    threshold: float
    severity: str  # "error", "warning"
    category: str  # "global", "per_engine", "per_language", "per_difficulty"
    message: str = ""


@dataclass
class BaselineComparison:
    """Comparison of current results against baseline snapshot."""
    engine: str
    metric: str
    baseline_value: float
    current_value: float
    change: float  # positive = degradation
    change_percent: float
    max_regression: float = 0.05  # 5% max regression allowed


@dataclass
class CIResult:
    """Complete CI check result."""
    passed: bool = True
    violations: list[ThresholdViolation] = field(default_factory=list)
    baseline_comparisons: list[BaselineComparison] = field(default_factory=list)
    summary: str = ""


class ThresholdChecker:
    """Checks benchmark results against CI thresholds and baselines."""

    def __init__(
        self,
        thresholds_path: str | None = None,
        baselines_path: str | None = None,
    ):
        """
        Initialize threshold checker.

        Args:
            thresholds_path: Path to thresholds.yaml. Defaults to config/thresholds.yaml.
            baselines_path: Path to baselines.yaml. Defaults to config/baselines.yaml.
        """
        base_config = Path(__file__).parent.parent.parent / "config"

        self.thresholds_path = Path(thresholds_path or base_config / "thresholds.yaml")
        self.baselines_path = Path(baselines_path or base_config / "baselines.yaml")

        self.thresholds: dict = {}
        self.baselines: dict = {}

        if self.thresholds_path.exists():
            with open(self.thresholds_path, encoding="utf-8") as f:
                self.thresholds = yaml.safe_load(f) or {}

        if self.baselines_path.exists():
            with open(self.baselines_path, encoding="utf-8") as f:
                self.baselines = yaml.safe_load(f) or {}

    def get_threshold(
        self,
        engine: str,
        metric: str,
        category: str = "global",
        subcategory: str = "",
    ) -> float | None:
        """Get threshold value for a specific engine/metric/category."""
        # Try engine-specific first
        engine_config = self.thresholds.get("engines", {}).get(engine, {})
        if metric in engine_config:
            return engine_config[metric]

        # Try category-specific
        if category in ("language", "difficulty"):
            cat_config = self.thresholds.get(f"{category}s", {}).get(subcategory, {})
            if metric in cat_config:
                return cat_config[metric]

        if category == "noise":
            cat_config = self.thresholds.get("noise_levels", {}).get(subcategory, {})
            if metric in cat_config:
                return cat_config[metric]

        # Fall back to global
        global_config = self.thresholds.get("global", {})
        return global_config.get(metric)

    def check_thresholds(
        self,
        engine_results: dict[str, EngineResult],
    ) -> list[ThresholdViolation]:
        """Check all engine results against thresholds."""
        violations = []

        for engine_name, result in engine_results.items():
            # Check global metrics
            self._check_metric(
                violations, engine_name, "cer", result.avg_cer, "global"
            )
            self._check_metric(
                violations, engine_name, "wer", result.avg_wer, "global"
            )
            self._check_metric_lower(
                violations, engine_name, "medical_accuracy",
                result.avg_medical_accuracy, "global"
            )
            self._check_metric_lower(
                violations, engine_name, "throughput",
                result.throughput, "global"
            )
            self._check_metric(
                violations, engine_name, "latency",
                result.avg_latency, "global"
            )

            # Check per-language metrics
            for lang, metrics in result.per_language.items():
                self._check_metric(
                    violations, engine_name, "cer",
                    metrics.get("cer", 0), "per_language", lang
                )
                self._check_metric(
                    violations, engine_name, "wer",
                    metrics.get("wer", 0), "per_language", lang
                )

            # Check per-difficulty metrics
            for diff, metrics in result.per_difficulty.items():
                self._check_metric(
                    violations, engine_name, "cer",
                    metrics.get("cer", 0), "per_difficulty", diff
                )

        return violations

    def _check_metric(
        self,
        violations: list[ThresholdViolation],
        engine: str,
        metric: str,
        value: float,
        category: str,
        subcategory: str = "",
    ) -> None:
        """Check if a metric exceeds its upper threshold."""
        max_metric = f"max_{metric}"
        threshold = self.get_threshold(engine, max_metric, category, subcategory)

        if threshold is not None and value > threshold:
            violations.append(ThresholdViolation(
                engine=engine,
                metric=metric,
                value=value,
                threshold=threshold,
                severity="error" if value > threshold * 1.5 else "warning",
                category=category,
                message=f"{engine} {metric}={value:.4f} exceeds threshold {threshold}",
            ))

    def _check_metric_lower(
        self,
        violations: list[ThresholdViolation],
        engine: str,
        metric: str,
        value: float,
        category: str,
        subcategory: str = "",
    ) -> None:
        """Check if a metric falls below its lower threshold."""
        min_metric = f"min_{metric}"
        threshold = self.get_threshold(engine, min_metric, category, subcategory)

        if threshold is not None and value < threshold:
            violations.append(ThresholdViolation(
                engine=engine,
                metric=metric,
                value=value,
                threshold=threshold,
                severity="error" if value < threshold * 0.7 else "warning",
                category=category,
                message=f"{engine} {metric}={value:.4f} below threshold {threshold}",
            ))

    def compare_against_baselines(
        self,
        engine_results: dict[str, EngineResult],
        max_regression: float = 0.05,
    ) -> list[BaselineComparison]:
        """Compare current results against baseline snapshots."""
        comparisons = []
        baselines_engines = self.baselines.get("engines", {})

        for engine_name, result in engine_results.items():
            baseline_engine = baselines_engines.get(engine_name, {})
            baseline_overall = baseline_engine.get("overall", {})

            # Compare overall metrics
            for metric_key, baseline_key in [
                ("avg_cer", "cer"),
                ("avg_wer", "wer"),
                ("avg_medical_accuracy", "medical_accuracy"),
            ]:
                baseline_value = baseline_overall.get(baseline_key, 0)
                current_value = getattr(result, metric_key, 0)

                if baseline_value > 0:
                    change = current_value - baseline_value
                    change_percent = (change / baseline_value) * 100

                    comparisons.append(BaselineComparison(
                        engine=engine_name,
                        metric=baseline_key,
                        baseline_value=baseline_value,
                        current_value=current_value,
                        change=change,
                        change_percent=change_percent,
                        max_regression=max_regression,
                    ))

            # Compare per-language
            baseline_langs = baseline_engine.get("english", {})
            if "english" in result.per_language:
                current = result.per_language["english"]
                for metric in ["cer", "wer"]:
                    baseline_val = baseline_langs.get(metric, 0)
                    current_val = current.get(metric, 0)
                    if baseline_val > 0:
                        change = current_val - baseline_val
                        change_percent = (change / baseline_val) * 100
                        comparisons.append(BaselineComparison(
                            engine=engine_name,
                            metric=f"english_{metric}",
                            baseline_value=baseline_val,
                            current_value=current_val,
                            change=change,
                            change_percent=change_percent,
                            max_regression=max_regression,
                        ))

        return comparisons

    def check(
        self,
        engine_results: dict[str, EngineResult],
        max_regression: float = 0.05,
        fail_on_warning: bool = False,
    ) -> CIResult:
        """
        Run full CI check against thresholds and baselines.

        Args:
            engine_results: Dict of engine name -> EngineResult
            max_regression: Maximum allowed regression percentage (default 5%)
            fail_on_warning: Whether to fail on warnings too

        Returns:
            CIResult with pass/fail status and details
        """
        result = CIResult()

        # Check thresholds
        violations = self.check_thresholds(engine_results)
        result.violations = violations

        # Check baselines
        baseline_comparisons = self.compare_against_baselines(
            engine_results, max_regression
        )
        result.baseline_comparisons = baseline_comparisons

        # Determine pass/fail
        errors = [v for v in violations if v.severity == "error"]
        regressions = [
            c for c in baseline_comparisons
            if c.change_percent > c.max_regression * 100
            and c.metric in ("cer", "wer")  # Only fail on CER/WER regression
        ]

        result.passed = (
            len(errors) == 0
            and (len(regressions) == 0 or not fail_on_warning)
        )

        # Build summary
        summary_parts = []
        if result.passed:
            summary_parts.append("✅ CI PASSED")
        else:
            summary_parts.append("❌ CI FAILED")

        if errors:
            summary_parts.append(f"  {len(errors)} threshold error(s):")
            for e in errors[:5]:
                summary_parts.append(f"    - {e.message}")

        if regressions:
            summary_parts.append(f"  {len(regressions)} baseline regression(s):")
            for r in regressions[:5]:
                summary_parts.append(
                    f"    - {r.engine} {r.metric}: {r.baseline_value:.4f} → "
                    f"{r.current_value:.4f} ({r.change_percent:+.1f}%)"
                )

        if warnings := [v for v in violations if v.severity == "warning"]:
            summary_parts.append(f"  {len(warnings)} warning(s)")

        result.summary = "\n".join(summary_parts)

        return result

    def print_ci_result(self, ci_result: CIResult) -> None:
        """Print CI result in a format suitable for GitHub Actions."""
        print(ci_result.summary)

        # Add GitHub Actions annotations if in CI
        if os.environ.get("GITHUB_ACTIONS"):
            for violation in ci_result.violations:
                level = "error" if violation.severity == "error" else "warning"
                print(f"::{level}::{violation.message}")

        if not ci_result.passed:
            sys.exit(1)
