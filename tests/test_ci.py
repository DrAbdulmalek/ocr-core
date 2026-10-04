"""Tests for CI threshold checking."""

import pytest
import yaml

from ocr_core.benchmarks.ci_suite.ci import CIResult, ThresholdChecker, ThresholdViolation
from ocr_core.benchmarks.ci_suite.report import EngineResult


@pytest.fixture
def thresholds_file(tmp_path):
    """Create a temporary thresholds.yaml file."""
    config = {
        "global": {
            "max_cer": 0.15,
            "max_wer": 0.25,
            "min_medical_accuracy": 0.80,
            "min_throughput": 0.5,
            "max_latency": 30.0,
        },
        "engines": {
            "test_engine": {
                "max_cer": 0.10,
                "max_wer": 0.20,
            }
        },
        "languages": {
            "english": {"max_cer": 0.08},
            "arabic": {"max_cer": 0.18},
        },
        "difficulty": {
            "easy": {"max_cer": 0.05},
            "hard": {"max_cer": 0.20},
        },
    }
    filepath = tmp_path / "thresholds.yaml"
    with open(filepath, "w") as f:
        yaml.dump(config, f)
    return str(filepath)


@pytest.fixture
def baselines_file(tmp_path):
    """Create a temporary baselines.yaml file."""
    config = {
        "version": "1.0",
        "engines": {
            "test_engine": {
                "overall": {
                    "cer": 0.10,
                    "wer": 0.18,
                    "medical_accuracy": 0.85,
                }
            }
        }
    }
    filepath = tmp_path / "baselines.yaml"
    with open(filepath, "w") as f:
        yaml.dump(config, f)
    return str(filepath)


@pytest.fixture
def good_engine_result():
    """Engine results that pass all thresholds."""
    return EngineResult(
        engine_name="test_engine",
        total_cases=10,
        avg_cer=0.05,
        avg_wer=0.10,
        avg_medical_accuracy=0.90,
        avg_latency=5.0,
        throughput=2.0,
        per_language={
            "english": {"cer": 0.03, "wer": 0.05, "medical_accuracy": 0.95},
            "arabic": {"cer": 0.08, "wer": 0.15, "medical_accuracy": 0.85},
        },
        per_difficulty={
            "easy": {"cer": 0.02, "wer": 0.03, "medical_accuracy": 0.98},
            "hard": {"cer": 0.10, "wer": 0.18, "medical_accuracy": 0.82},
        },
    )


@pytest.fixture
def bad_engine_result():
    """Engine results that fail thresholds."""
    return EngineResult(
        engine_name="test_engine",
        total_cases=10,
        avg_cer=0.25,  # Exceeds max_cer 0.15
        avg_wer=0.40,  # Exceeds max_wer 0.25
        avg_medical_accuracy=0.60,  # Below min 0.80
        avg_latency=50.0,  # Exceeds max 30.0
        throughput=0.1,  # Below min 0.5
        per_language={
            "english": {"cer": 0.15, "wer": 0.25, "medical_accuracy": 0.70},
        },
    )


class TestThresholdChecker:
    def test_init_loads_config(self, thresholds_file, baselines_file):
        checker = ThresholdChecker(thresholds_file, baselines_file)
        assert checker.thresholds is not None
        assert "global" in checker.thresholds
        assert checker.baselines is not None

    def test_init_missing_files(self, tmp_path):
        checker = ThresholdChecker(
            str(tmp_path / "nonexistent_t.yaml"),
            str(tmp_path / "nonexistent_b.yaml"),
        )
        assert checker.thresholds == {}
        assert checker.baselines == {}

    def test_get_threshold_global(self, thresholds_file, baselines_file):
        checker = ThresholdChecker(thresholds_file, baselines_file)
        assert checker.get_threshold("any_engine", "max_cer", "global") == 0.15

    def test_get_threshold_engine_specific(self, thresholds_file, baselines_file):
        checker = ThresholdChecker(thresholds_file, baselines_file)
        assert checker.get_threshold("test_engine", "max_cer", "global") == 0.10

    def test_get_threshold_language(self, thresholds_file, baselines_file):
        checker = ThresholdChecker(thresholds_file, baselines_file)
        assert checker.get_threshold("any", "max_cer", "language", "english") == 0.08

    def test_get_threshold_not_found(self, thresholds_file, baselines_file):
        checker = ThresholdChecker(thresholds_file, baselines_file)
        assert checker.get_threshold("any", "max_nonexistent", "global") is None

    def test_check_thresholds_passing(self, thresholds_file, baselines_file, good_engine_result):
        checker = ThresholdChecker(thresholds_file, baselines_file)
        violations = checker.check_thresholds({"test_engine": good_engine_result})
        assert len(violations) == 0

    def test_check_thresholds_failing(self, thresholds_file, baselines_file, bad_engine_result):
        checker = ThresholdChecker(thresholds_file, baselines_file)
        violations = checker.check_thresholds({"test_engine": bad_engine_result})
        assert len(violations) > 0
        # Should have CER, WER, medical_accuracy, latency, throughput violations
        metric_names = {v.metric for v in violations}
        assert "cer" in metric_names
        assert "wer" in metric_names
        assert "medical_accuracy" in metric_names

    def test_compare_against_baselines(self, thresholds_file, baselines_file, good_engine_result):
        checker = ThresholdChecker(thresholds_file, baselines_file)
        comparisons = checker.compare_against_baselines(
            {"test_engine": good_engine_result}
        )
        assert len(comparisons) > 0
        # All comparisons should show improvement (negative change)
        for comp in comparisons:
            if comp.metric in ("cer", "wer"):
                assert comp.change <= 0  # Better than baseline

    def test_check_passed(self, thresholds_file, baselines_file, good_engine_result):
        checker = ThresholdChecker(thresholds_file, baselines_file)
        result = checker.check({"test_engine": good_engine_result})
        assert result.passed is True
        assert "PASSED" in result.summary

    def test_check_failed(self, thresholds_file, baselines_file, bad_engine_result):
        checker = ThresholdChecker(thresholds_file, baselines_file)
        result = checker.check({"test_engine": bad_engine_result})
        assert result.passed is False
        assert "FAILED" in result.summary

    def test_check_no_engines(self, thresholds_file, baselines_file):
        checker = ThresholdChecker(thresholds_file, baselines_file)
        result = checker.check({})
        assert result.passed is True

    def test_violation_severity_error(self, thresholds_file, baselines_file):
        checker = ThresholdChecker(thresholds_file, baselines_file)
        # CER 0.30 >> 0.15 threshold -> should be error (1.5x)
        result = EngineResult(
            engine_name="test_engine",
            avg_cer=0.30,
        )
        violations = checker.check_thresholds({"test_engine": result})
        cer_violations = [v for v in violations if v.metric == "cer"]
        assert len(cer_violations) > 0
        assert cer_violations[0].severity == "error"

    def test_violation_severity_warning(self, thresholds_file, baselines_file):
        checker = ThresholdChecker(thresholds_file, baselines_file)
        # Use engine without specific threshold to use global (max_cer: 0.15)
        # CER 0.18 > 0.15 but < 0.225 -> should be warning
        result = EngineResult(
            engine_name="other_engine",
            avg_cer=0.18,
        )
        violations = checker.check_thresholds({"other_engine": result})
        cer_violations = [v for v in violations if v.metric == "cer"]
        assert len(cer_violations) > 0
        assert cer_violations[0].severity == "warning"


class TestCIResult:
    def test_default_passed(self):
        result = CIResult()
        assert result.passed is True
        assert result.violations == []

    def test_with_violations(self):
        result = CIResult(
            passed=False,
            violations=[
                ThresholdViolation(
                    engine="test",
                    metric="cer",
                    value=0.30,
                    threshold=0.15,
                    severity="error",
                    category="global",
                    message="CER exceeded",
                )
            ],
        )
        assert result.passed is False
        assert len(result.violations) == 1
