"""Unified OCR Benchmark Metrics.

Provides CER, WER, medical term accuracy, latency, and throughput metrics
for the medical OCR ecosystem benchmark suite.
"""

import statistics
import time


class EditDistance:
    """Levenshtein edit distance with backtrace for CER/WER."""

    @staticmethod
    def distance(s1: str, s2: str) -> int:
        """Compute Levenshtein distance between two strings.

        Uses the classic dynamic programming algorithm with O(n*m) time
        and O(min(n,m)) space complexity.

        Args:
            s1: Reference string.
            s2: Hypothesis string.

        Returns:
            The minimum number of single-character edits (insertions,
            deletions, substitutions) required to change s1 into s2.
        """
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost is 0 if characters match, 1 otherwise
                cost = 0 if c1 == c2 else 1
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + cost
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def _alignment(s1: str, s2: str) -> tuple[list, list]:
        """Compute full DP matrix and backtrace alignment between two strings.

        Args:
            s1: Reference string.
            s2: Hypothesis string.

        Returns:
            A tuple of (operations, counts) where operations is a list of
            operation labels and counts is a dict with 'S', 'D', 'I' keys.
        """
        n, m = len(s1), len(s2)

        # Build full DP matrix
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1],  # substitution
                    )

        # Backtrace to count S, D, I
        substitutions = 0
        deletions = 0
        insertions = 0

        i, j = n, m
        while i > 0 and j > 0:
            if s1[i - 1] == s2[j - 1]:
                # Correct / match — no operation needed
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j - 1] + 1:
                # Substitution
                substitutions += 1
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j] + 1:
                # Deletion from reference
                deletions += 1
                i -= 1
            else:
                # Insertion (extra in hypothesis)
                insertions += 1
                j -= 1

        # Handle remaining characters
        deletions += i  # remaining chars in reference were deleted
        insertions += j  # remaining chars in hypothesis were inserted

        return [], {
            "S": substitutions,
            "D": deletions,
            "I": insertions,
        }

    @staticmethod
    def cer(reference: str, hypothesis: str) -> dict:
        """Character Error Rate with S, D, I breakdown.

        CER = (S + D + I) / len(reference)

        Args:
            reference: Ground truth text.
            hypothesis: OCR output text.

        Returns:
            A dict with keys: 'cer' (float), 'substitutions' (int),
            'deletions' (int), 'insertions' (int), 'ref_length' (int).
        """
        if not reference:
            if hypothesis:
                return {
                    "cer": float("inf") if len(hypothesis) > 0 else 0.0,
                    "substitutions": 0,
                    "deletions": 0,
                    "insertions": len(hypothesis),
                    "ref_length": 0,
                }
            return {
                "cer": 0.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "ref_length": 0,
            }

        _, counts = EditDistance._alignment(reference, hypothesis)
        ref_length = len(reference)
        total_errors = counts["S"] + counts["D"] + counts["I"]

        return {
            "cer": total_errors / ref_length,
            "substitutions": counts["S"],
            "deletions": counts["D"],
            "insertions": counts["I"],
            "ref_length": ref_length,
        }

    @staticmethod
    def wer(reference: str, hypothesis: str) -> dict:
        """Word Error Rate with S, D, I breakdown.

        WER = (S + D + I) / len(reference_words)

        Args:
            reference: Ground truth text.
            hypothesis: OCR output text.

        Returns:
            A dict with keys: 'wer' (float), 'substitutions' (int),
            'deletions' (int), 'insertions' (int), 'ref_length' (int).
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        if not ref_words:
            if hyp_words:
                return {
                    "wer": float("inf"),
                    "substitutions": 0,
                    "deletions": 0,
                    "insertions": len(hyp_words),
                    "ref_length": 0,
                }
            return {
                "wer": 0.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "ref_length": 0,
            }

        _, counts = EditDistance._alignment(
            " ".join(ref_words),
            " ".join(hyp_words),
        )
        ref_length = len(ref_words)
        total_errors = counts["S"] + counts["D"] + counts["I"]

        return {
            "wer": total_errors / ref_length,
            "substitutions": counts["S"],
            "deletions": counts["D"],
            "insertions": counts["I"],
            "ref_length": ref_length,
        }


class LatencyProfiler:
    """Measure and report processing latency.

    Runs a function multiple times with optional warmup, collecting
    timing statistics for benchmarking purposes.
    """

    def __init__(self, warmup_runs: int = 3, benchmark_runs: int = 10):
        """Initialize the latency profiler.

        Args:
            warmup_runs: Number of warmup iterations (not measured).
            benchmark_runs: Number of measured iterations.
        """
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self._timings: list[float] = []

    def measure(self, func, *args, **kwargs) -> dict:
        """Run function multiple times and return timing statistics.

        Performs warmup runs first, then runs the function
        ``benchmark_runs`` times, recording the wall-clock time for each.

        Args:
            func: Callable to benchmark.
            *args: Positional arguments passed to func.
            **kwargs: Keyword arguments passed to func.

        Returns:
            A dict with keys: 'mean' (float, seconds), 'median' (float),
            'stdev' (float), 'min' (float), 'max' (float),
            'p95' (float), 'p99' (float), 'runs' (int),
            'total' (float).
        """
        # Warmup phase — not recorded
        for _ in range(self.warmup_runs):
            func(*args, **kwargs)

        # Benchmark phase
        timings = []
        for _ in range(self.benchmark_runs):
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

        self._timings.extend(timings)

        sorted_t = sorted(timings)
        n = len(sorted_t)

        def percentile(data: list[float], p: float) -> float:
            """Compute the p-th percentile from sorted data."""
            if n == 1:
                return data[0]
            k = (p / 100.0) * (n - 1)
            f = int(k)
            c = f + 1 if f + 1 < n else f
            d = k - f
            return sorted_t[f] + d * (sorted_t[c] - sorted_t[f])

        return {
            "mean": statistics.mean(timings),
            "median": statistics.median(timings),
            "stdev": statistics.stdev(timings) if n > 1 else 0.0,
            "min": min(timings),
            "max": max(timings),
            "p95": percentile(sorted_t, 95),
            "p99": percentile(sorted_t, 99),
            "runs": n,
            "total": sum(timings),
        }

    def summary(self) -> dict:
        """Return aggregated latency statistics across all measured runs.

        Returns:
            A dict with the same structure as ``measure`` but aggregating
            all runs that have been measured so far.
        """
        if not self._timings:
            return {
                "mean": 0.0,
                "median": 0.0,
                "stdev": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "runs": 0,
                "total": 0.0,
            }

        sorted_t = sorted(self._timings)
        n = len(sorted_t)

        def percentile(data: list[float], p: float) -> float:
            if n == 1:
                return data[0]
            k = (p / 100.0) * (n - 1)
            f = int(k)
            c = f + 1 if f + 1 < n else f
            d = k - f
            return sorted_t[f] + d * (sorted_t[c] - sorted_t[f])

        return {
            "mean": statistics.mean(self._timings),
            "median": statistics.median(self._timings),
            "stdev": statistics.stdev(self._timings) if n > 1 else 0.0,
            "min": min(self._timings),
            "max": max(self._timings),
            "p95": percentile(sorted_t, 95),
            "p99": percentile(sorted_t, 99),
            "runs": n,
            "total": sum(self._timings),
        }

    def reset(self) -> None:
        """Clear all recorded timings."""
        self._timings.clear()


class MedicalTermEvaluator:
    """Evaluate how well medical terms are preserved in OCR output.

    Checks whether medical terms from a reference list appear (exactly or
    approximately) in the hypothesis text.
    """

    def __init__(self, medical_terms: list[str]):
        """Initialize with a list of expected medical terms.

        Args:
            medical_terms: List of medical terms to check for in output.
        """
        self.medical_terms = [t.strip().lower() for t in medical_terms if t.strip()]

    def evaluate(self, reference: str, hypothesis: str) -> dict:
        """Check medical term preservation accuracy.

        For each medical term, checks if it appears in the hypothesis text.
        Supports exact match and approximate match (edit distance ≤ 2
        or ratio ≥ 0.85).

        Args:
            reference: Ground truth text (unused directly, but available
                for context).
            hypothesis: OCR output text to evaluate.

        Returns:
            A dict with keys:
                'total_terms' (int): Number of terms evaluated.
                'found_exact' (int): Exact matches found.
                'found_approximate' (int): Approximate matches (not exact).
                'missing' (int): Terms not found at all.
                'accuracy' (float): (exact + approximate) / total.
                'exact_accuracy' (float): exact / total.
                'details' (list[dict]): Per-term breakdown.
        """
        hypothesis_lower = hypothesis.lower()
        details = []
        found_exact = 0
        found_approximate = 0
        missing = 0

        for term in self.medical_terms:
            if term in hypothesis_lower:
                found_exact += 1
                details.append({
                    "term": term,
                    "status": "exact",
                    "distance": 0,
                })
            else:
                # Try approximate matching: check if any subsequence of
                # words in the hypothesis has low edit distance
                best_distance = len(term)
                hyp_words = hypothesis_lower.split()
                ref_words = term.split()

                # Check each sliding window in hypothesis words
                for window_size in range(max(1, len(ref_words) - 1), min(len(ref_words) + 2, len(hyp_words) + 1)):
                    for start in range(len(hyp_words) - window_size + 1):
                        window = " ".join(hyp_words[start:start + window_size])
                        d = EditDistance.distance(term, window)
                        best_distance = min(best_distance, d)

                # Also check character-level distance
                char_dist = EditDistance.distance(term, hypothesis_lower[:len(term)])
                best_distance = min(best_distance, char_dist)

                ratio = 1.0 - (best_distance / max(len(term), 1))
                if best_distance <= 2 or ratio >= 0.85:
                    found_approximate += 1
                    details.append({
                        "term": term,
                        "status": "approximate",
                        "distance": best_distance,
                        "similarity_ratio": round(ratio, 4),
                    })
                else:
                    missing += 1
                    details.append({
                        "term": term,
                        "status": "missing",
                        "distance": best_distance,
                        "similarity_ratio": round(ratio, 4),
                    })

        total = len(self.medical_terms)
        found_total = found_exact + found_approximate

        return {
            "total_terms": total,
            "found_exact": found_exact,
            "found_approximate": found_approximate,
            "missing": missing,
            "accuracy": found_total / total if total > 0 else 0.0,
            "exact_accuracy": found_exact / total if total > 0 else 0.0,
            "details": details,
        }


class BenchmarkSuite:
    """Run a complete benchmark suite and generate reports.

    Collects individual benchmark results and provides aggregated summaries.
    """

    def __init__(self, name: str):
        """Initialize a benchmark suite.

        Args:
            name: Descriptive name for this benchmark suite.
        """
        self.name = name
        self.results: list[dict] = []

    def add_result(self, benchmark_name: str, metrics: dict, metadata: dict = None) -> None:
        """Add a benchmark result to the suite.

        Args:
            benchmark_name: Name/identifier of the benchmark.
            metrics: Dict of metric names to values (e.g. {'cer': 0.05}).
            metadata: Optional dict with additional context (engine, model, etc.).
        """
        entry = {
            "benchmark_name": benchmark_name,
            "metrics": metrics,
            "metadata": metadata or {},
        }
        self.results.append(entry)

    def summary(self) -> dict:
        """Aggregate all results into a summary.

        Computes statistics for each metric across all benchmark results.

        Returns:
            A dict with keys:
                'suite_name' (str): The suite name.
                'total_benchmarks' (int): Number of benchmarks run.
                'metrics_summary' (dict): Per-metric aggregation.
                'results' (list[dict]): Raw results.
        """
        if not self.results:
            return {
                "suite_name": self.name,
                "total_benchmarks": 0,
                "metrics_summary": {},
                "results": [],
            }

        # Collect all metric keys across results
        all_metric_keys: set[str] = set()
        for entry in self.results:
            all_metric_keys.update(entry["metrics"].keys())

        metrics_summary: dict[str, dict] = {}
        for key in sorted(all_metric_keys):
            values = []
            for entry in self.results:
                val = entry["metrics"].get(key)
                if val is not None and isinstance(val, (int, float)):
                    values.append(float(val))

            if values:
                metrics_summary[key] = {
                    "mean": round(statistics.mean(values), 6),
                    "median": round(statistics.median(values), 6),
                    "stdev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
                    "min": round(min(values), 6),
                    "max": round(max(values), 6),
                    "count": len(values),
                }
            else:
                metrics_summary[key] = {"values": [], "count": 0}

        return {
            "suite_name": self.name,
            "total_benchmarks": len(self.results),
            "metrics_summary": metrics_summary,
            "results": self.results,
        }
