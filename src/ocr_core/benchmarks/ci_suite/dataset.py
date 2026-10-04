"""
dataset.py — Dataset management for medical OCR benchmarks.
إدارة مجموعة البيانات لمعايير تقييم OCR الطبي.

Provides:
- Load and index test cases from JSON files
- Filter by language, specialty, difficulty, noise level
- Split datasets for cross-validation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestCase:
    """A single benchmark test case with ground truth and metadata."""
    id: str
    language: str  # "english", "arabic", "mixed"
    specialty: str  # "cardiology", "radiology", "prescriptions", etc.
    difficulty: str  # "easy", "medium", "hard"
    noise_level: str  # "clean", "light_noise", "moderate_noise", "heavy_noise"
    image_path: str  # Path to the test image (may be placeholder)
    source: str  # "synthetic", "real", "contributed"
    description: str  # Human-readable description
    ground_truth: str  # The expected OCR output
    extra_metadata: dict = field(default_factory=dict)


@dataclass
class DatasetStats:
    """Statistics about a dataset."""
    total_cases: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    specialties: dict[str, int] = field(default_factory=dict)
    difficulties: dict[str, int] = field(default_factory=dict)
    noise_levels: dict[str, int] = field(default_factory=dict)
    total_characters: int = 0
    total_words: int = 0


class DatasetManager:
    """Manages benchmark test case datasets."""

    def __init__(self, data_dir: str | None = None):
        """
        Initialize dataset manager.

        Args:
            data_dir: Path to the data directory. Defaults to data/ in package root.
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.cases: list[TestCase] = []
        self._index: dict[str, TestCase] = {}

    def load(self) -> DatasetManager:
        """Load all test cases from the data directory."""
        self.cases = []
        self._index = {}

        for lang_dir in ["english", "arabic", "mixed"]:
            lang_path = self.data_dir / lang_dir
            if not lang_path.exists():
                continue

            for json_file in sorted(lang_path.glob("*.json")):
                try:
                    with open(json_file, encoding="utf-8") as f:
                        data = json.load(f)
                    case = TestCase(
                        id=data.get("id", json_file.stem),
                        language=data.get("language", lang_dir),
                        specialty=data.get("specialty", "unknown"),
                        difficulty=data.get("difficulty", "medium"),
                        noise_level=data.get("noise_level", "clean"),
                        image_path=data.get("image_path", ""),
                        source=data.get("source", "unknown"),
                        description=data.get("description", ""),
                        ground_truth=data.get("ground_truth", ""),
                        extra_metadata={
                            k: v for k, v in data.items()
                            if k not in {
                                "id", "language", "specialty", "difficulty",
                                "noise_level", "image_path", "source",
                                "description", "ground_truth",
                            }
                        },
                    )
                    self.cases.append(case)
                    self._index[case.id] = case
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Failed to load {json_file}: {e}")

        return self

    def get_case(self, case_id: str) -> TestCase | None:
        """Get a specific test case by ID."""
        return self._index.get(case_id)

    def get_stats(self) -> DatasetStats:
        """Calculate statistics about the loaded dataset."""
        stats = DatasetStats(total_cases=len(self.cases))

        for case in self.cases:
            stats.languages[case.language] = stats.languages.get(case.language, 0) + 1
            stats.specialties[case.specialty] = stats.specialties.get(case.specialty, 0) + 1
            stats.difficulties[case.difficulty] = stats.difficulties.get(case.difficulty, 0) + 1
            stats.noise_levels[case.noise_level] = stats.noise_levels.get(case.noise_level, 0) + 1

            # Count characters and words in ground truth
            gt = case.ground_truth.strip()
            stats.total_characters += len(gt)
            stats.total_words += len(gt.split())

        return stats

    def filter(
        self,
        language: str | None = None,
        specialty: str | None = None,
        difficulty: str | None = None,
        noise_level: str | None = None,
        exclude_languages: set[str] | None = None,
    ) -> list[TestCase]:
        """
        Filter test cases by metadata.

        Args:
            language: Filter by language ("english", "arabic", "mixed")
            specialty: Filter by medical specialty
            difficulty: Filter by difficulty level
            noise_level: Filter by noise level
            exclude_languages: Exclude specific languages

        Returns:
            Filtered list of TestCase objects
        """
        cases = self.cases

        if language:
            cases = [c for c in cases if c.language == language]

        if specialty:
            cases = [c for c in cases if c.specialty == specialty]

        if difficulty:
            cases = [c for c in cases if c.difficulty == difficulty]

        if noise_level:
            cases = [c for c in cases if c.noise_level == noise_level]

        if exclude_languages:
            cases = [c for c in cases if c.language not in exclude_languages]

        return cases

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
        stratify_by: str | None = None,
    ) -> tuple:
        """
        Split dataset into train/test sets.

        Args:
            train_ratio: Ratio of training set
            seed: Random seed for reproducibility
            stratify_by: Field to stratify by (e.g., "language", "specialty")

        Returns:
            Tuple of (train_cases, test_cases)
        """
        import random

        random.seed(seed)

        if stratify_by:
            groups: dict[str, list[TestCase]] = {}
            for case in self.cases:
                key = getattr(case, stratify_by, "unknown")
                groups.setdefault(key, []).append(case)

            train_cases = []
            test_cases = []
            for group in groups.values():
                random.shuffle(group)
                split_idx = int(len(group) * train_ratio)
                train_cases.extend(group[:split_idx])
                test_cases.extend(group[split_idx:])
        else:
            shuffled = list(self.cases)
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * train_ratio)
            train_cases = shuffled[:split_idx]
            test_cases = shuffled[split_idx:]

        return train_cases, test_cases

    def add_case(self, case: TestCase) -> None:
        """Add a new test case to the dataset."""
        self.cases.append(case)
        self._index[case.id] = case

    def remove_case(self, case_id: str) -> bool:
        """Remove a test case by ID. Returns True if removed."""
        if case_id in self._index:
            del self._index[case_id]
            self.cases = [c for c in self.cases if c.id != case_id]
            return True
        return False

    def get_unique_values(self, field_name: str) -> list[str]:
        """Get unique values for a given field across all cases."""
        values = set()
        for case in self.cases:
            if hasattr(case, field_name):
                values.add(getattr(case, field_name))
        return sorted(values)

    def to_dataframe(self) -> pandas.DataFrame:
        """Export dataset as a pandas DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame([
                {
                    "id": c.id,
                    "language": c.language,
                    "specialty": c.specialty,
                    "difficulty": c.difficulty,
                    "noise_level": c.noise_level,
                    "source": c.source,
                    "description": c.description,
                    "ground_truth_length": len(c.ground_truth),
                    "word_count": len(c.ground_truth.split()),
                }
                for c in self.cases
            ])
        except ImportError:
            raise ImportError("pandas is required for DataFrame export. Install with: pip install pandas")

    def __len__(self) -> int:
        return len(self.cases)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"DatasetManager({len(self.cases)} cases, "
            f"languages={stats.languages}, "
            f"specialties={list(stats.specialties.keys())})"
        )
