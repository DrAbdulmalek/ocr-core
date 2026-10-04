"""Tests for report generation."""
import json

import pytest

from ocr_core.benchmarks.ci_suite.report import (
    BenchmarkReport,
    EngineResult,
    ReportGenerator,
)


@pytest.fixture
def sample_engine_results():
    """Sample engine results for testing."""
    return {
        "engine_a": EngineResult(
            engine_name="engine_a",
            total_cases=10,
            avg_cer=0.08,
            avg_wer=0.15,
            avg_medical_accuracy=0.88,
            avg_latency=5.0,
            throughput=2.0,
            per_language={
                "english": {"cer": 0.05, "wer": 0.10, "medical_accuracy": 0.92},
                "arabic": {"cer": 0.12, "wer": 0.20, "medical_accuracy": 0.84},
            },
            per_specialty={
                "cardiology": {"cer": 0.06, "wer": 0.12, "medical_accuracy": 0.90},
            },
            per_difficulty={
                "easy": {"cer": 0.03, "wer": 0.05, "medical_accuracy": 0.95},
                "hard": {"cer": 0.15, "wer": 0.25, "medical_accuracy": 0.80},
            },
        ),
        "engine_b": EngineResult(
            engine_name="engine_b",
            total_cases=10,
            avg_cer=0.12,
            avg_wer=0.22,
            avg_medical_accuracy=0.82,
            avg_latency=10.0,
            throughput=1.0,
            per_language={
                "english": {"cer": 0.08, "wer": 0.15, "medical_accuracy": 0.87},
                "arabic": {"cer": 0.18, "wer": 0.30, "medical_accuracy": 0.77},
            },
            per_specialty={},
            per_difficulty={},
        ),
    }


@pytest.fixture
def sample_report(sample_engine_results):
    """Sample benchmark report for testing."""
    return BenchmarkReport(
        timestamp="2024-01-15T10:30:00",
        duration=45.2,
        engine_results=sample_engine_results,
        metadata={"total_cases": 10, "engines": ["engine_a", "engine_b"]},
    )


class TestReportGenerator:
    @pytest.fixture
    def generator(self, tmp_path):
        return ReportGenerator(str(tmp_path))

    def test_generate_json(self, generator, sample_report):
        path = generator.generate_json(sample_report)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "engines" in data
        assert "engine_a" in data["engines"]
        assert data["engines"]["engine_a"]["avg_cer"] == 0.08

    def test_generate_markdown(self, generator, sample_report):
        path = generator.generate_markdown(sample_report)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "# Medical OCR Benchmark" in content
        assert "engine_a" in content
        assert "By Language" in content

    def test_generate_html(self, generator, sample_report):
        path = generator.generate_html(sample_report)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "<table>" in content
        assert "engine_a" in content

    def test_generate_all(self, generator, sample_report):
        paths = generator.generate_all(sample_report, ["json", "markdown", "html"])
        assert len(paths) == 3
        for path in paths:
            assert path.exists()

    def test_invalid_format(self, generator, sample_report):
        with pytest.raises(ValueError, match="Unknown format"):
            generator.generate_all(sample_report, ["pdf"])

    def test_output_dir_created(self, tmp_path):
        new_dir = tmp_path / "new_reports"
        ReportGenerator(str(new_dir))
        assert new_dir.exists()

    def test_json_roundtrip(self, generator, sample_report):
        path = generator.generate_json(sample_report)
        with open(path) as f:
            data = json.load(f)
        # Verify all engines are present
        assert len(data["engines"]) == 2


class TestEngineResult:
    def test_default_values(self):
        result = EngineResult(engine_name="test")
        assert result.total_cases == 0
        assert result.avg_cer == 0.0
        assert result.case_results == []

    def test_per_language_empty(self):
        result = EngineResult(engine_name="test")
        assert result.per_language == {}


class TestBenchmarkReport:
    def test_default_values(self):
        report = BenchmarkReport()
        assert report.timestamp == ""
        assert report.engine_results == {}
        assert report.duration == 0.0
