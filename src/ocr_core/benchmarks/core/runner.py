"""Benchmark Runner — Orchestrates benchmark execution.

Loads golden datasets, dispatches to individual benchmark implementations,
and aggregates results.
"""

import json
import os
from typing import Any

from ocr_core.benchmarks.core.metrics import BenchmarkSuite


class BenchmarkRunner:
    """Orchestrates execution of benchmarks against golden datasets.

    Supports running individual benchmarks or the full suite, with optional
    comparison of two result sets.
    """

    def __init__(self, config_path: str = None):
        """Initialize the benchmark runner.

        Args:
            config_path: Optional path to a JSON configuration file. If
                not provided, sensible defaults are used.
        """
        self.config = self._load_config(config_path)
        self.benchmarks: dict[str, Any] = {}
        self._suite = BenchmarkSuite(name="medical-ocr-benchmark-suite")
        self._register_defaults()

    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from JSON file or return defaults.

        Args:
            config_path: Path to a JSON config file.

        Returns:
            A dict of configuration options.
        """
        defaults = {
            "golden_dir": "data/golden",
            "output_dir": "results",
            "warmup_runs": 3,
            "benchmark_runs": 10,
            "benchmarks": ["correction", "phi"],
        }

        if config_path and os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as f:
                user_config = json.load(f)
            defaults.update(user_config)

        return defaults

    def _register_defaults(self) -> None:
        """Register built-in benchmark implementations."""
        self.benchmarks["correction"] = {
            "module": "benchmarks.postprocessor.correction",
            "class": "CorrectionBenchmark",
            "description": "Postprocessor correction accuracy benchmark",
        }
        self.benchmarks["phi"] = {
            "module": "benchmarks.postprocessor.phi",
            "class": "PHIBenchmark",
            "description": "PHI detection and masking benchmark",
        }

    def register(self, name: str, module: str, cls: str, description: str = "") -> None:
        """Register a custom benchmark implementation.

        Args:
            name: Unique name for this benchmark.
            module: Python module path (e.g. 'benchmarks.ocr.tesseract').
            cls: Class name within the module.
            description: Human-readable description.
        """
        self.benchmarks[name] = {
            "module": module,
            "class": cls,
            "description": description,
        }

    def _load_benchmark_instance(self, name: str):
        """Dynamically load a benchmark class and instantiate it.

        Args:
            name: Registered benchmark name.

        Returns:
            An instance of the benchmark class.

        Raises:
            ValueError: If the benchmark name is not registered.
            ImportError: If the module or class cannot be found.
        """
        if name not in self.benchmarks:
            available = ", ".join(sorted(self.benchmarks.keys()))
            raise ValueError(
                f"Unknown benchmark '{name}'. Available: {available}"
            )

        spec = self.benchmarks[name]
        module_path = spec["module"]
        class_name = spec["class"]

        try:
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls()
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"Cannot load benchmark '{name}': {module_path}.{class_name} — {exc}"
            ) from exc

    def _load_golden_dataset(self, dataset_name: str = "en_medical") -> dict:
        """Load a golden dataset from the configured directory.

        Args:
            dataset_name: Name of the dataset file (without .json extension).

        Returns:
            The loaded dataset dict.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
        """
        golden_dir = self.config["golden_dir"]
        filename = dataset_name if dataset_name.endswith(".json") else f"{dataset_name}.json"
        filepath = os.path.join(golden_dir, filename)

        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Golden dataset not found: {filepath}"
            )

        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    def run_single(self, benchmark_name: str, dataset_name: str = "en_medical") -> dict:
        """Run a single benchmark against a golden dataset.

        Args:
            benchmark_name: Name of the registered benchmark to run.
            dataset_name: Name of the golden dataset to use.

        Returns:
            A dict with keys: 'benchmark_name', 'dataset', 'metrics', 'metadata'.
        """
        benchmark_instance = self._load_benchmark_instance(benchmark_name)
        golden_dataset = self._load_golden_dataset(dataset_name)

        if hasattr(benchmark_instance, "run"):
            metrics = benchmark_instance.run(golden_dataset)
        else:
            metrics = {
                "error": f"Benchmark '{benchmark_name}' has no 'run' method",
            }

        result = {
            "benchmark_name": benchmark_name,
            "dataset": dataset_name,
            "metrics": metrics,
            "metadata": {
                "description": self.benchmarks.get(benchmark_name, {}).get("description", ""),
                "golden_file": dataset_name,
            },
        }

        # Add to suite for aggregation
        self._suite.add_result(
            benchmark_name=benchmark_name,
            metrics=metrics,
            metadata=result["metadata"],
        )

        return result

    def run_all(self, datasets: list[str] = None) -> dict:
        """Run all registered benchmarks against all specified datasets.

        Args:
            datasets: List of dataset names. If None, runs against all
                JSON files in the golden directory.

        Returns:
            A dict with keys: 'suite_name', 'config', 'results' (list),
            'summary'.
        """
        if datasets is None:
            # Auto-discover datasets in golden directory
            golden_dir = self.config["golden_dir"]
            if os.path.isdir(golden_dir):
                datasets = [
                    f[:-5]  # strip .json
                    for f in os.listdir(golden_dir)
                    if f.endswith(".json")
                ]
            else:
                datasets = ["en_medical"]

        all_results = []

        for benchmark_name in sorted(self.benchmarks.keys()):
            for dataset_name in datasets:
                try:
                    result = self.run_single(benchmark_name, dataset_name)
                    all_results.append(result)
                except (FileNotFoundError, ImportError, ValueError) as exc:
                    all_results.append({
                        "benchmark_name": benchmark_name,
                        "dataset": dataset_name,
                        "metrics": {"error": str(exc)},
                        "metadata": {},
                    })

        summary = self._suite.summary()

        return {
            "suite_name": self.config.get("suite_name", "medical-ocr-benchmark-suite"),
            "config": self.config,
            "results": all_results,
            "summary": summary,
        }

    def compare(self, results_a: dict, results_b: dict) -> dict:
        """Compare two benchmark results and compute deltas.

        Args:
            results_a: First benchmark result dict (must contain 'metrics').
            results_b: Second benchmark result dict (must contain 'metrics').

        Returns:
            A dict with keys:
                'benchmark_a' (str): Name/description of result A.
                'benchmark_b' (str): Name/description of result B.
                'comparison' (dict): Per-metric comparison with deltas.
                'winner' (str): Which result is better overall.
        """
        metrics_a = results_a.get("metrics", {})
        metrics_b = results_b.get("metrics", {})

        all_keys = set(metrics_a.keys()) | set(metrics_b.keys())
        comparison = {}
        wins_a = 0
        wins_b = 0

        for key in sorted(all_keys):
            val_a = metrics_a.get(key)
            val_b = metrics_b.get(key)

            entry: dict[str, Any] = {
                "metric": key,
                "value_a": val_a,
                "value_b": val_b,
            }

            if val_a is not None and val_b is not None:
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    delta = val_b - val_a
                    entry["delta"] = round(delta, 6)
                    entry["delta_percent"] = round(
                        (delta / val_a * 100) if val_a != 0 else float("inf"), 2
                    )

                    # For error-rate metrics (lower is better), A wins if B > A
                    if key.lower() in ("cer", "wer", "latency", "mean", "median"):
                        if val_a < val_b:
                            wins_a += 1
                        elif val_b < val_a:
                            wins_b += 1
                    else:
                        # For accuracy-like metrics (higher is better)
                        if val_a > val_b:
                            wins_a += 1
                        elif val_b > val_a:
                            wins_b += 1

            comparison[key] = entry

        if wins_a > wins_b:
            winner = results_a.get("benchmark_name", "A")
        elif wins_b > wins_a:
            winner = results_b.get("benchmark_name", "B")
        else:
            winner = "tie"

        return {
            "benchmark_a": results_a.get("benchmark_name", "A"),
            "benchmark_b": results_b.get("benchmark_name", "B"),
            "comparison": comparison,
            "winner": winner,
            "wins_a": wins_a,
            "wins_b": wins_b,
        }
