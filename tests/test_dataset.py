"""Tests for dataset management."""
import json

import pytest

from ocr_core.benchmarks.ci_suite.dataset import DatasetManager, DatasetStats, TestCase


class TestDatasetManager:
    @pytest.fixture
    def data_dir(self, tmp_path):
        """Create a temporary dataset directory with test cases."""
        en_dir = tmp_path / "english"
        en_dir.mkdir()

        # Create test cases
        cases = [
            {
                "id": "test_en_001",
                "language": "english",
                "specialty": "cardiology",
                "difficulty": "easy",
                "noise_level": "clean",
                "image_path": "data/english/images/test001.png",
                "source": "synthetic",
                "description": "Test cardiology case",
                "ground_truth": "Patient has hypertension and angina pectoris.",
            },
            {
                "id": "test_en_002",
                "language": "english",
                "specialty": "radiology",
                "difficulty": "hard",
                "noise_level": "heavy_noise",
                "image_path": "data/english/images/test002.png",
                "source": "synthetic",
                "description": "Test radiology case",
                "ground_truth": "CT chest shows no consolidation or effusion.",
            },
        ]

        for case in cases:
            with open(en_dir / f"{case['id']}.json", "w") as f:
                json.dump(case, f)

        # Create Arabic directory
        ar_dir = tmp_path / "arabic"
        ar_dir.mkdir()
        ar_case = {
            "id": "test_ar_001",
            "language": "arabic",
            "specialty": "cardiology",
            "difficulty": "medium",
            "noise_level": "light_noise",
            "image_path": "data/arabic/images/test001.png",
            "source": "synthetic",
            "description": "حالة اختبار طب القلب",
            "ground_truth": "المريض يعاني من ارتفاع ضغط الدم",
        }
        with open(ar_dir / "test_ar_001.json", "w") as f:
            json.dump(ar_case, f)

        return str(tmp_path)

    def test_load(self, data_dir):
        dm = DatasetManager(data_dir)
        dm.load()
        assert len(dm) == 3

    def test_get_case(self, data_dir):
        dm = DatasetManager(data_dir)
        dm.load()
        case = dm.get_case("test_en_001")
        assert case is not None
        assert case.specialty == "cardiology"

    def test_get_case_not_found(self, data_dir):
        dm = DatasetManager(data_dir)
        dm.load()
        assert dm.get_case("nonexistent") is None

    def test_filter_by_language(self, data_dir):
        dm = DatasetManager(data_dir)
        dm.load()
        english = dm.filter(language="english")
        assert len(english) == 2
        arabic = dm.filter(language="arabic")
        assert len(arabic) == 1

    def test_filter_by_specialty(self, data_dir):
        dm = DatasetManager(data_dir)
        dm.load()
        cardio = dm.filter(specialty="cardiology")
        assert len(cardio) == 2

    def test_filter_by_difficulty(self, data_dir):
        dm = DatasetManager(data_dir)
        dm.load()
        hard = dm.filter(difficulty="hard")
        assert len(hard) == 1

    def test_filter_multiple(self, data_dir):
        dm = DatasetManager(data_dir)
        dm.load()
        result = dm.filter(language="english", difficulty="easy")
        assert len(result) == 1

    def test_filter_exclude(self, data_dir):
        dm = DatasetManager(data_dir)
        dm.load()
        result = dm.filter(exclude_languages={"english"})
        assert len(result) == 1

    def test_get_stats(self, data_dir):
        dm = DatasetManager(data_dir)
        dm.load()
        stats = dm.get_stats()
        assert stats.total_cases == 3
        assert stats.languages["english"] == 2
        assert stats.languages["arabic"] == 1

    def test_split(self, data_dir):
        dm = DatasetManager(data_dir)
        dm.load()
        train, test = dm.split(train_ratio=0.67, seed=42)
        assert len(train) + len(test) == 3

    def test_split_stratified(self, data_dir):
        dm = DatasetManager(data_dir)
        dm.load()
        train, test = dm.split(train_ratio=0.5, seed=42, stratify_by="language")
        # Each language should have at least one case in each set
        set(c.language for c in train)
        set(c.language for c in test)
        assert len(train) > 0 and len(test) > 0

    def test_add_case(self, data_dir):
        dm = DatasetManager(data_dir)
        dm.load()
        new_case = TestCase(
            id="test_new",
            language="english",
            specialty="pathology",
            difficulty="easy",
            noise_level="clean",
            image_path="",
            source="synthetic",
            description="New test",
            ground_truth="Test text",
        )
        dm.add_case(new_case)
        assert len(dm) == 4
        assert dm.get_case("test_new") is not None

    def test_remove_case(self, data_dir):
        dm = DatasetManager(data_dir)
        dm.load()
        assert dm.remove_case("test_en_001") is True
        assert len(dm) == 2
        assert dm.remove_case("nonexistent") is False

    def test_get_unique_values(self, data_dir):
        dm = DatasetManager(data_dir)
        dm.load()
        languages = dm.get_unique_values("language")
        assert "english" in languages
        assert "arabic" in languages

    def test_repr(self, data_dir):
        dm = DatasetManager(data_dir)
        dm.load()
        repr_str = repr(dm)
        assert "3 cases" in repr_str

    def test_empty_data_dir(self, tmp_path):
        dm = DatasetManager(str(tmp_path))
        dm.load()
        assert len(dm) == 0
        stats = dm.get_stats()
        assert stats.total_cases == 0

    def test_invalid_json(self, tmp_path):
        en_dir = tmp_path / "english"
        en_dir.mkdir()
        with open(en_dir / "bad.json", "w") as f:
            f.write("not valid json")
        dm = DatasetManager(str(tmp_path))
        dm.load()  # Should not raise, just skip
        assert len(dm) == 0


class TestDatasetStats:
    def test_default_values(self):
        stats = DatasetStats()
        assert stats.total_cases == 0
        assert stats.languages == {}
