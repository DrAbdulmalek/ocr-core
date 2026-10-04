"""Golden Dataset Loader for benchmarks.

Provides loading, validation, and listing of golden datasets used
for benchmarking OCR and postprocessor performance.
"""

import json
import os


class GoldenDataset:
    """Load and validate golden benchmark datasets.

    Golden datasets contain test cases with reference texts, hypothesis
    texts (OCR output), medical terms, and optional metadata for evaluation.
    """

    def __init__(self, dataset_dir: str = "data/golden"):
        """Initialize the golden dataset loader.

        Args:
            dataset_dir: Path to the directory containing golden dataset
                JSON files.
        """
        self.dataset_dir = dataset_dir
        self._cache: dict[str, dict] = {}

    def load(self, name: str) -> dict:
        """Load a golden dataset by name.

        Args:
            name: Dataset name (with or without .json extension).

        Returns:
            The loaded dataset dict.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the dataset fails validation.
        """
        # Normalize name
        if not name.endswith(".json"):
            name = f"{name}.json"

        # Check cache
        if name in self._cache:
            return self._cache[name]

        filepath = os.path.join(self.dataset_dir, name)

        if not os.path.isfile(filepath):
            available = self.list_datasets()
            raise FileNotFoundError(
                f"Dataset '{name}' not found at '{filepath}'. "
                f"Available datasets: {', '.join(available) if available else 'none'}"
            )

        with open(filepath, encoding="utf-8") as f:
            dataset = json.load(f)

        # Validate
        if not self.validate(dataset):
            raise ValueError(f"Dataset '{name}' failed validation")

        self._cache[name] = dataset
        return dataset

    def list_datasets(self) -> list[str]:
        """List all available golden datasets in the dataset directory.

        Returns:
            A sorted list of dataset names (without .json extension).
        """
        if not os.path.isdir(self.dataset_dir):
            return []

        datasets = []
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith(".json"):
                datasets.append(filename[:-5])  # strip .json

        return sorted(datasets)

    def validate(self, dataset: dict) -> bool:
        """Validate a golden dataset structure.

        Checks that the dataset has the required fields and that each
        test case has the minimum required data.

        Args:
            dataset: A dataset dict to validate.

        Returns:
            True if the dataset is valid, False otherwise.
        """
        if not isinstance(dataset, dict):
            return False

        # Check for required top-level fields
        if "name" not in dataset:
            return False

        if "test_cases" not in dataset:
            return False

        test_cases = dataset["test_cases"]
        if not isinstance(test_cases, list):
            return False

        if len(test_cases) == 0:
            return True  # Empty but valid

        # Validate each test case
        for i, case in enumerate(test_cases):
            if not isinstance(case, dict):
                return False

            # Required fields per test case
            if "id" not in case:
                return False

            if "reference" not in case:
                return False

            if "hypothesis" not in case:
                return False

            # Validate types
            if not isinstance(case["reference"], str):
                return False

            if not isinstance(case["hypothesis"], str):
                return False

            # medical_terms is optional but should be a list if present
            if "medical_terms" in case:
                if not isinstance(case["medical_terms"], list):
                    return False

            # ocr_confidence is optional but should be a number if present
            if "ocr_confidence" in case:
                if not isinstance(case["ocr_confidence"], (int, float)):
                    return False
                if not (0.0 <= case["ocr_confidence"] <= 1.0):
                    return False

        return True

    def get_test_case(self, dataset_name: str, case_id: str) -> dict:
        """Get a specific test case from a dataset.

        Args:
            dataset_name: Name of the dataset.
            case_id: ID of the test case to retrieve.

        Returns:
            The test case dict.

        Raises:
            KeyError: If the test case ID is not found.
        """
        dataset = self.load(dataset_name)
        for case in dataset.get("test_cases", []):
            if case.get("id") == case_id:
                return case

        available_ids = [c.get("id") for c in dataset.get("test_cases", [])]
        raise KeyError(
            f"Test case '{case_id}' not found in dataset '{dataset_name}'. "
            f"Available IDs: {', '.join(available_ids)}"
        )

    def summary(self, dataset_name: str) -> dict:
        """Get a summary of a golden dataset.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            A dict with summary statistics.
        """
        dataset = self.load(dataset_name)
        test_cases = dataset.get("test_cases", [])

        categories: dict[str, int] = {}
        total_terms = 0
        total_ref_chars = 0
        total_hyp_chars = 0
        confidences = []

        for case in test_cases:
            cat = case.get("category", "uncategorized")
            categories[cat] = categories.get(cat, 0) + 1

            medical_terms = case.get("medical_terms", [])
            total_terms += len(medical_terms)
            total_ref_chars += len(case.get("reference", ""))
            total_hyp_chars += len(case.get("hypothesis", ""))

            conf = case.get("ocr_confidence")
            if conf is not None:
                confidences.append(conf)

        return {
            "name": dataset.get("name", dataset_name),
            "language": dataset.get("language", "unknown"),
            "total_cases": len(test_cases),
            "categories": categories,
            "total_medical_terms": total_terms,
            "total_reference_chars": total_ref_chars,
            "total_hypothesis_chars": total_hyp_chars,
            "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
            "min_confidence": round(min(confidences), 4) if confidences else None,
            "max_confidence": round(max(confidences), 4) if confidences else None,
        }
