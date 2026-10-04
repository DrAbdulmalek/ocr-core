"""Tests for the Threshold Checker module."""

import json
import sys
from pathlib import Path


from ocr_core.benchmarks.ci_suite.threshold_checker import (
    BenchmarkResult,
    DecisionEngine,
    ReportGenerator,
    ResultsReader,
    SignalEmitter,
    ThresholdConfig,
)


class TestBenchmarkResult:
    def test_creation(self):
        r = BenchmarkResult(
            model_name="test-model", test_type="printed",
            cer=0.03, wer=0.05, timestamp="2026-06-25T00:00:00", dataset_size=100
        )
        assert r.model_name == "test-model"
        assert r.test_type == "printed"
        assert r.cer == 0.03


class TestResultsReader:
    def test_from_json(self, tmp_path):
        data = [
            {"model_name": "m1", "test_type": "printed", "cer": 0.03, "wer": 0.05,
             "timestamp": "2026-06-25", "dataset_size": 100}
        ]
        f = tmp_path / "results.json"
        f.write_text(json.dumps(data))
        results = ResultsReader.from_json(str(f))
        assert len(results) == 1
        assert results[0].cer == 0.03

    def test_from_json_with_defaults(self, tmp_path):
        data = [{"model_name": "m1"}]
        f = tmp_path / "results.json"
        f.write_text(json.dumps(data))
        results = ResultsReader.from_json(str(f))
        assert results[0].cer == 0.0
        assert results[0].dataset_size == 0


class TestDecisionEngine:
    def setup_method(self):
        self.config = ThresholdConfig()
        self.engine = DecisionEngine(self.config)

    def test_printed_passes(self):
        result = BenchmarkResult("m1", "printed", 0.04, 0.08, "2026-06-25", 100)
        all_passed, decisions = self.engine.evaluate([result])
        assert all_passed is True
        assert decisions[0]['action'] == 'DEPLOY'

    def test_printed_fails_cer(self):
        result = BenchmarkResult("m1", "printed", 0.06, 0.08, "2026-06-25", 100)
        all_passed, decisions = self.engine.evaluate([result])
        assert all_passed is False
        assert decisions[0]['action'] == 'RETRAIN'

    def test_handwritten_passes(self):
        result = BenchmarkResult("m1", "handwritten", 0.10, 0.15, "2026-06-25", 100)
        all_passed, decisions = self.engine.evaluate([result])
        assert all_passed is True

    def test_handwritten_fails_cer(self):
        result = BenchmarkResult("m1", "handwritten", 0.15, 0.15, "2026-06-25", 100)
        all_passed, decisions = self.engine.evaluate([result])
        assert all_passed is False

    def test_mixed_results(self):
        results = [
            BenchmarkResult("m1", "printed", 0.04, 0.08, "2026-06-25", 100),
            BenchmarkResult("m1", "handwritten", 0.15, 0.15, "2026-06-25", 100),
        ]
        all_passed, _ = self.engine.evaluate(results)
        assert all_passed is False


class TestReportGenerator:
    def test_generate_passed_report(self):
        decisions = [{
            'model_name': 'm1', 'test_type': 'printed', 'cer': 0.04,
            'cer_threshold': 0.05, 'cer_passed': True,
            'wer': 0.08, 'wer_threshold': 0.10, 'wer_passed': True,
            'passed': True, 'action': 'DEPLOY', 'timestamp': '2026-06-25'
        }]
        report = ReportGenerator.generate(decisions, True)
        assert 'PASSED' in report
        assert 'DEPLOY' in report

    def test_generate_failed_report(self):
        decisions = [{
            'model_name': 'm1', 'test_type': 'printed', 'cer': 0.06,
            'cer_threshold': 0.05, 'cer_passed': False,
            'wer': 0.08, 'wer_threshold': 0.10, 'wer_passed': True,
            'passed': False, 'action': 'RETRAIN', 'timestamp': '2026-06-25'
        }]
        report = ReportGenerator.generate(decisions, False)
        assert 'FAILED' in report
        assert 'RETRAIN' in report

    def test_report_with_ab_result(self):
        decisions = [{
            'model_name': 'm1', 'test_type': 'printed', 'cer': 0.04,
            'cer_threshold': 0.05, 'cer_passed': True,
            'wer': 0.08, 'wer_threshold': 0.10, 'wer_passed': True,
            'passed': True, 'action': 'DEPLOY', 'timestamp': '2026-06-25'
        }]
        ab_result = {
            'current_cer': 0.048, 'new_cer': 0.042,
            'improvement_percent': 12.5, 'decision': 'DEPLOY',
            'reason': 'Model improved by 12.5%'
        }
        report = ReportGenerator.generate(decisions, True, ab_result)
        assert 'A/B Test Result' in report
        assert '12.5%' in report


class TestSignalEmitter:
    def test_emit_deploy_signal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        SignalEmitter.emit_deploy_signal()
        assert (tmp_path / '.deploy_signal').read_text() == 'DEPLOY_APPROVED'

    def test_emit_retrain_signal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        SignalEmitter.emit_retrain_signal()
        assert (tmp_path / '.retrain_signal').read_text() == 'RETRAIN_REQUIRED'


class TestThresholdConfig:
    def test_custom_config(self):
        config = ThresholdConfig(printed_cer_threshold=0.03, handwritten_cer_threshold=0.10)
        assert config.printed_cer_threshold == 0.03
        assert config.handwritten_cer_threshold == 0.10

    def test_default_config(self):
        config = ThresholdConfig()
        assert config.printed_cer_threshold == 0.05
        assert config.handwritten_cer_threshold == 0.12
