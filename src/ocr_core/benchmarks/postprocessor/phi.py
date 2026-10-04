"""Benchmark PHI detection and masking performance.

Tests Protected Health Information detection accuracy, masking
performance, and throughput across different modes.
"""

import re
import statistics

from ocr_core.benchmarks.core.metrics import LatencyProfiler

# Common PHI patterns used for fallback detection
_PHI_PATTERNS: list[dict] = [
    {
        "name": "patient_name",
        "pattern": re.compile(
            r"(?:patient\s+name|name)\s*[:\-]?\s*(.+?)(?:\n|$)",
            re.IGNORECASE,
        ),
        "label": "PATIENT_NAME",
    },
    {
        "name": "date_of_birth",
        "pattern": re.compile(
            r"(?:date\s+of\s+birth|dob|تاريخ\s+الميلاد|DOB)\s*[:\-]?\s*(.+?)(?:\n|$)",
            re.IGNORECASE,
        ),
        "label": "DATE_OF_BIRTH",
    },
    {
        "name": "phone_number",
        "pattern": re.compile(
            r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b",
        ),
        "label": "PHONE",
    },
    {
        "name": "email",
        "pattern": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        ),
        "label": "EMAIL",
    },
    {
        "name": "medical_record_number",
        "pattern": re.compile(
            r"\b(?:MRN|mrn)\s*[:\-]?\s*\d{4,}\b",
        ),
        "label": "MRN",
    },
    {
        "name": "date_pattern",
        "pattern": re.compile(
            r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b",
        ),
        "label": "DATE",
    },
]


class PHIBenchmark:
    """Benchmark PHI detection and masking performance.

    Detects and masks Protected Health Information in medical text,
    measuring accuracy and throughput against golden annotations.
    """

    def __init__(self):
        """Initialize the PHI benchmark."""
        self._postprocessor = None
        self._available: bool | None = None

    def _is_available(self) -> bool:
        """Check if the postprocessor PHI module is available.

        Returns:
            True if the postprocessor can do PHI detection, False otherwise.
        """
        if self._available is not None:
            return self._available

        try:
            from medical_ocr_postprocessor import MedicalOCRPostprocessor
            self._postprocessor = MedicalOCRPostprocessor()
            self._available = hasattr(self._postprocessor, "mask_phi") or hasattr(self._postprocessor, "detect_phi")
        except ImportError:
            self._available = False

        return self._available

    def _detect_phi(self, text: str) -> list[dict]:
        """Detect PHI entities in text.

        Uses the postprocessor if available, otherwise falls back to
        regex-based detection.

        Args:
            text: Text to scan for PHI.

        Returns:
            A list of dicts with 'entity_type', 'text', 'start', 'end' keys.
        """
        if self._is_available() and self._postprocessor is not None:
            try:
                if hasattr(self._postprocessor, "detect_phi"):
                    return self._postprocessor.detect_phi(text)
                if hasattr(self._postprocessor, "find_phi"):
                    return self._postprocessor.find_phi(text)
            except Exception:
                pass

        # Fallback: regex-based PHI detection
        entities = []
        for phi_def in _PHI_PATTERNS:
            for match in phi_def["pattern"].finditer(text):
                entities.append({
                    "entity_type": phi_def["label"],
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })

        return entities

    def _mask_phi(self, text: str, mode: str = "tag") -> str:
        """Mask PHI entities in text.

        Args:
            text: Text containing potential PHI.
            mode: Masking mode — 'tag' replaces with [ENTITY_TYPE],
                'replace' replaces with '***', 'redact' removes entirely.

        Returns:
            Text with PHI masked.
        """
        entities = self._detect_phi(text)

        # Sort by start position in reverse order to maintain indices
        sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)

        result = text
        for entity in sorted_entities:
            entity_text = entity["text"]
            if mode == "tag":
                replacement = f"[{entity['entity_type']}]"
            elif mode == "replace":
                replacement = "*" * len(entity_text)
            elif mode == "redact":
                replacement = ""
            else:
                replacement = f"[{entity['entity_type']}]"

            result = result[:entity["start"]] + replacement + result[entity["end"]:]

        return result

    def benchmark_detection(self, texts: list[str]) -> dict:
        """Benchmark PHI detection accuracy and speed.

        Args:
            texts: List of medical texts containing PHI.

        Returns:
            A dict with detection counts, latencies, and coverage metrics.
        """
        if not texts:
            return {"error": "Empty text list"}

        profiler = LatencyProfiler(warmup_runs=1, benchmark_runs=5)

        def detect_all():
            return [self._detect_phi(t) for t in texts]

        timing = profiler.measure(detect_all)
        all_entities = detect_all()

        entity_counts = [len(e) for e in all_entities]
        all_entity_types = []
        for entities in all_entities:
            for e in entities:
                if isinstance(e, dict):
                    all_entity_types.append(e.get("entity_type", "unknown"))

        type_counts: dict[str, int] = {}
        for et in all_entity_types:
            type_counts[et] = type_counts.get(et, 0) + 1

        return {
            "total_texts": len(texts),
            "total_entities_found": sum(entity_counts),
            "mean_entities_per_text": round(statistics.mean(entity_counts), 2) if entity_counts else 0,
            "max_entities_in_text": max(entity_counts) if entity_counts else 0,
            "min_entities_in_text": min(entity_counts) if entity_counts else 0,
            "entity_type_breakdown": type_counts,
            "postprocessor_available": self._is_available(),
            "latency": timing,
        }

    def benchmark_masking(self, texts: list[str], mode: str = "tag") -> dict:
        """Benchmark PHI masking performance across different modes.

        Args:
            texts: List of medical texts containing PHI.
            mode: Masking mode ('tag', 'replace', or 'redact').

        Returns:
            A dict with masking throughput, latency, and coverage metrics.
        """
        if not texts:
            return {"error": "Empty text list"}

        if mode not in ("tag", "replace", "redact"):
            return {"error": f"Invalid mode '{mode}'. Use 'tag', 'replace', or 'redact'."}

        profiler = LatencyProfiler(warmup_runs=1, benchmark_runs=5)

        def mask_all():
            return [self._mask_phi(t, mode=mode) for t in texts]

        timing = profiler.measure(mask_all)
        masked_texts = mask_all()

        # Count entities that were masked
        total_masked = 0
        masked_texts_with_content = []
        for original, masked in zip(texts, masked_texts):
            entities = self._detect_phi(original)
            total_masked += len(entities)
            if masked != original:
                masked_texts_with_content.append({
                    "original": original,
                    "masked": masked,
                })

        total_chars = sum(len(t) for t in texts)
        throughput = total_chars / timing["mean"] if timing["mean"] > 0 else 0

        return {
            "mode": mode,
            "total_texts": len(texts),
            "total_entities_masked": total_masked,
            "texts_modified": len(masked_texts_with_content),
            "total_chars_processed": total_chars,
            "throughput_chars_per_sec": round(throughput, 2),
            "postprocessor_available": self._is_available(),
            "latency": timing,
        }

    def run(self, golden_dataset: dict) -> dict:
        """Run PHI benchmark against a golden dataset.

        Processes each test case for PHI detection and masking,
        measuring accuracy and performance.

        Args:
            golden_dataset: A golden dataset dict with 'test_cases' key.

        Returns:
            A dict with aggregate PHI benchmark metrics.
        """
        test_cases = golden_dataset.get("test_cases", [])
        if not test_cases:
            return {
                "engine": "phi",
                "status": "error",
                "error": "No test cases in golden dataset",
            }

        texts = [case.get("hypothesis", "") for case in test_cases]

        # Detection benchmark
        detection_result = self.benchmark_detection(texts)

        # Masking benchmark in all modes
        masking_results = {}
        for mode in ("tag", "replace", "redact"):
            masking_results[mode] = self.benchmark_masking(texts, mode=mode)

        return {
            "engine": "phi",
            "status": "success",
            "total_cases": len(test_cases),
            "postprocessor_available": self._is_available(),
            "detection": {
                "total_entities_found": detection_result.get("total_entities_found", 0),
                "mean_entities_per_text": detection_result.get("mean_entities_per_text", 0),
                "entity_type_breakdown": detection_result.get("entity_type_breakdown", {}),
                "mean_latency_s": detection_result.get("latency", {}).get("mean"),
            },
            "masking": {
                mode: {
                    "total_entities_masked": r.get("total_entities_masked", 0),
                    "texts_modified": r.get("texts_modified", 0),
                    "throughput_chars_per_sec": r.get("throughput_chars_per_sec", 0),
                    "mean_latency_s": r.get("latency", {}).get("mean"),
                }
                for mode, r in masking_results.items()
            },
        }
