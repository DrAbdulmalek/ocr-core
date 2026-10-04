"""Tests for the A/B Tester module."""

import json
import sys
from pathlib import Path


from ocr_core.benchmarks.ci_suite.ab_testing import ABTester, HoldoutManager, create_sample_holdout


class TestHoldoutManager:
    def test_load_holdout(self, tmp_path):
        data = {
            'created': '2026-06-25',
            'source': 'test',
            'test_ratio': 0.2,
            'seed': 42,
            'total_original': 10,
            'train_size': 8,
            'holdout_size': 2,
            'samples': [
                {'image_path': 'img1.png', 'ground_truth': 'text1', 'ocr_engine': 'auto'},
                {'image_path': 'img2.png', 'ground_truth': 'text2', 'ocr_engine': 'auto'}
            ]
        }
        f = tmp_path / "holdout.json"
        f.write_text(json.dumps(data))
        mgr = HoldoutManager(str(f))
        assert mgr.size == 2

    def test_create_from_dataset(self, tmp_path):
        # Create a source dataset
        dataset = [
            {'image_path': f'img{i}.png', 'ground_truth': f'text {i}'}
            for i in range(10)
        ]
        src = tmp_path / "full_dataset.json"
        src.write_text(json.dumps(dataset))

        holdout_path = tmp_path / "holdout.json"
        mgr = HoldoutManager(str(holdout_path))
        count = mgr.create_from_dataset(str(src), test_ratio=0.2)

        assert count == 2
        assert mgr.size == 2
        assert (tmp_path / "full_dataset_train.json").exists()
        assert holdout_path.exists()


class TestABTester:
    def test_improvement_deploy(self, tmp_path):
        holdout = {
            'created': '2026-06-25', 'source': 'test',
            'samples': [{'image_path': 'img.png', 'ground_truth': 'text', 'ocr_engine': 'auto'}]
        }
        f = tmp_path / "holdout.json"
        f.write_text(json.dumps(holdout))
        tester = ABTester(str(f))
        result = tester.run_ab_test(0.048, 0.042)
        assert result['decision'] == 'DEPLOY'
        assert result['improvement_percent'] > 0

    def test_regression_rollback(self, tmp_path):
        holdout = {
            'created': '2026-06-25', 'source': 'test',
            'samples': [{'image_path': 'img.png', 'ground_truth': 'text', 'ocr_engine': 'auto'}]
        }
        f = tmp_path / "holdout.json"
        f.write_text(json.dumps(holdout))
        tester = ABTester(str(f))
        result = tester.run_ab_test(0.048, 0.055)
        assert result['decision'] == 'ROLLBACK'
        assert result['improvement_percent'] < 0

    def test_first_model_deploy(self, tmp_path):
        holdout = {
            'created': '2026-06-25', 'source': 'test',
            'samples': [{'image_path': 'img.png', 'ground_truth': 'text', 'ocr_engine': 'auto'}]
        }
        f = tmp_path / "holdout.json"
        f.write_text(json.dumps(holdout))
        tester = ABTester(str(f))
        result = tester.run_ab_test(0.0, 0.05)
        assert result['decision'] == 'DEPLOY'
        assert 'First model' in result['reason']

    def test_save_history(self, tmp_path):
        holdout = {
            'created': '2026-06-25', 'source': 'test',
            'samples': [{'image_path': 'img.png', 'ground_truth': 'text', 'ocr_engine': 'auto'}]
        }
        f = tmp_path / "holdout.json"
        f.write_text(json.dumps(holdout))
        tester = ABTester(str(f))
        tester.run_ab_test(0.048, 0.042)
        history_path = str(tmp_path / "history.json")
        tester.save_history(history_path)
        assert Path(history_path).exists()
        data = json.loads(Path(history_path).read_text())
        assert len(data) == 1


class TestCreateSampleHoldout:
    def test_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = create_sample_holdout()
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert 'samples' in data
        assert len(data['samples']) == 2
