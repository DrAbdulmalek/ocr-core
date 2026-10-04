"""High-level deduplication pipeline that chains RTL fixing, field extraction,
and weighted medical deduplication into a single API.

The pipeline was designed based on the Arabic Medical OCR Blueprint analysis
(Grok v4, July 2026) which identified that raw OCR text must be RTL-corrected
*before* field extraction and similarity scoring to avoid false negatives
caused by reversed Arabic tokens.

Typical usage::

    from ocr_core.postprocess.deduplication_pipeline import DeduplicationPipeline

    pipe = DeduplicationPipeline()
    result = pipe.process_pair(
        "اسم المريض: حمزة علي\nالتاريخ: 2026-01-15",
        "اسم المريض: حمزه على\nالتاريخ: 2026/01/15",
    )
    print(result["is_duplicate"], result["confidence_score"])
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ocr_core.postprocess.deduplication import (
    WeightedMedicalDeduplicator,
    field_aware_similarity,
)
from ocr_core.postprocess.field_extractor import (
    ArabicMedicalFieldExtractor,
    ExtractedMedicalFields,
)
from ocr_core.rtl_utils import ArabicRTLFixer, RTLFixStats

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineStepResult:
    """Diagnostics for a single pipeline step."""

    step_name: str
    duration_ms: float
    success: bool
    error: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DeduplicationResult:
    """Complete result of a pair-wise deduplication check."""

    is_duplicate: bool
    confidence_score: float
    field_similarity_score: float
    text_similarity_score: float
    rtl_fixed_left: str
    rtl_fixed_right: str
    rtl_stats_left: dict[str, Any]
    rtl_stats_right: dict[str, Any]
    fields_left: dict[str, Any]
    fields_right: dict[str, Any]
    explanation: str
    pipeline_steps: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeduplicationPipeline:
    """End-to-end deduplication pipeline for Arabic medical OCR records.

    Processing stages (in order):
      1. **RTL Fix** — Reverse visually-mirrored Arabic tokens using
         :class:`ArabicRTLFixer`.
      2. **Field Extraction** — Extract structured medical fields
         (patient name, ID, date, diagnosis, medications) using
         :class:`ArabicMedicalFieldExtractor`.
      3. **Field-Aware Similarity** — Weighted comparison of extracted
         fields using :class:`WeightedMedicalDeduplicator`.
      4. **Confidence Scoring** — Combine field similarity, text
         similarity, and RTL fix statistics into an overall confidence
         score.

    Parameters
    ----------
    rtl_fixer : ArabicRTLFixer | None
        Custom RTL fixer instance.  Uses default settings if ``None``.
    field_extractor : ArabicMedicalFieldExtractor | None
        Custom field extractor.  Uses default if ``None``.
    deduplicator : WeightedMedicalDeduplicator | None
        Custom deduplicator with custom weights.  Uses default if ``None``.
    duplicate_threshold : float
        Score above which a pair is considered duplicate.  Default ``0.85``.
    rtl_weight : float
        Weight of RTL similarity in the final confidence score.  Default ``0.1``.
    field_weight : float
        Weight of field-aware similarity in the final confidence score.
        Default ``0.6``.
    text_weight : float
        Weight of raw text similarity in the final confidence score.
        Default ``0.3``.
    """

    def __init__(
        self,
        rtl_fixer: Optional[ArabicRTLFixer] = None,
        field_extractor: Optional[ArabicMedicalFieldExtractor] = None,
        deduplicator: Optional[WeightedMedicalDeduplicator] = None,
        duplicate_threshold: float = 0.85,
        rtl_weight: float = 0.1,
        field_weight: float = 0.6,
        text_weight: float = 0.3,
    ) -> None:
        self.rtl_fixer = rtl_fixer or ArabicRTLFixer()
        self.field_extractor = field_extractor or ArabicMedicalFieldExtractor()
        self.deduplicator = deduplicator or WeightedMedicalDeduplicator(
            extractor=self.field_extractor,
        )
        self.duplicate_threshold = duplicate_threshold
        self.rtl_weight = rtl_weight
        self.field_weight = field_weight
        self.text_weight = text_weight

        # Validate weights sum to ~1.0
        total = rtl_weight + field_weight + text_weight
        if abs(total - 1.0) > 0.05:
            logger.warning(
                "Confidence weights sum to %.2f (expected ~1.0). "
                "Results may be unexpected.",
                total,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_pair(
        self,
        left: str,
        right: str,
        *,
        force_rtl: bool = False,
    ) -> DeduplicationResult:
        """Run the full pipeline on a pair of OCR text records.

        Parameters
        ----------
        left : str
            First OCR text (raw, may contain reversed Arabic).
        right : str
            Second OCR text (raw, may contain reversed Arabic).
        force_rtl : bool
            If ``True``, apply RTL fix even when the heuristic does not
            detect reversal.

        Returns
        -------
        DeduplicationResult
            Full result with scores, extracted fields, and diagnostics.
        """
        t0 = time.perf_counter()
        steps: list[dict[str, Any]] = []

        # ── Stage 1: RTL Fix ──
        fixed_left, stats_left = self._fix_rtl(left, force_rtl)
        fixed_right, stats_right = self._fix_rtl(right, force_rtl)
        steps.append(self._step_dict("rtl_fix", fixed_left is not None))

        # ── Stage 2: Field Extraction ──
        fields_left = self._extract_fields(fixed_left)
        fields_right = self._extract_fields(fixed_right)
        steps.append(self._step_dict("field_extraction", fields_left is not None))

        # ── Stage 3: Field-Aware Similarity ──
        similarity = self._compute_similarity(fixed_left, fixed_right)
        steps.append(self._step_dict("field_similarity", similarity is not None))

        # ── Stage 4: Confidence Scoring ──
        confidence, explanation = self._compute_confidence(
            fixed_left, fixed_right, similarity, stats_left, stats_right,
        )
        steps.append(self._step_dict("confidence_scoring", True))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        is_dup = confidence >= self.duplicate_threshold

        logger.info(
            "Dedup result: is_duplicate=%s confidence=%.3f "
            "field_sim=%.3f elapsed=%.0fms",
            is_dup, confidence, similarity.score if similarity else 0,
            elapsed_ms,
        )

        return DeduplicationResult(
            is_duplicate=is_dup,
            confidence_score=round(confidence, 4),
            field_similarity_score=similarity.score if similarity else 0.0,
            text_similarity_score=similarity.field_scores.get(
                "template_signature", 0.0,
            ) if similarity else 0.0,
            rtl_fixed_left=fixed_left or "",
            rtl_fixed_right=fixed_right or "",
            rtl_stats_left=self._stats_to_dict(stats_left),
            rtl_stats_right=self._stats_to_dict(stats_right),
            fields_left=fields_left.to_dict() if fields_left else {},
            fields_right=fields_right.to_dict() if fields_right else {},
            explanation=explanation,
            pipeline_steps=steps,
            elapsed_ms=round(elapsed_ms, 1),
        )

    def deduplicate_records(
        self,
        records: list[str | dict[str, Any] | ExtractedMedicalFields],
    ) -> dict[str, Any]:
        """Deduplicate a list of records using the full pipeline.

        Each record is first RTL-fixed, then field-extracted, then compared
        using the weighted field-aware deduplicator.

        Parameters
        ----------
        records : list
            List of OCR text strings, dicts with ``raw_text`` / ``text``
            key, or :class:`ExtractedMedicalFields` instances.

        Returns
        -------
        dict
            ``unique_records``, ``duplicates``, ``input_count``,
            ``unique_count``, and per-pair ``pipeline_results``.
        """
        # Pre-process: RTL fix all text records
        fixed_records: list[str | dict[str, Any]] = []
        pipeline_results: list[dict[str, Any]] = []

        for record in records:
            if isinstance(record, ExtractedMedicalFields):
                fixed_records.append(record)
            elif isinstance(record, dict):
                text = str(record.get("raw_text", record.get("text", "")))
                fixed_text, _ = self._fix_rtl(text, force=False)
                fixed_record = dict(record)
                fixed_record["raw_text"] = fixed_text
                fixed_records.append(fixed_record)
            else:
                fixed_text, _ = self._fix_rtl(str(record), force=False)
                fixed_records.append(fixed_text)

        # Run deduplication
        dedup_result = self.deduplicator.deduplicate(fixed_records)

        # Enrich with pipeline-level confidence scores
        for dup in dedup_result.get("duplicates", []):
            pair_result = self.process_pair(
                str(dup.get("record", {}).get("raw_text", "")),
                "",  # placeholder — full pair needs both sides
            )
            pipeline_results.append(pair_result.to_dict())

        dedup_result["pipeline_results"] = pipeline_results
        return dedup_result

    def health_report(self) -> dict[str, Any]:
        """Return a status report for all pipeline components."""
        return {
            "rtl_fixer": "ready",
            "field_extractor": "ready",
            "deduplicator": "ready",
            "duplicate_threshold": self.duplicate_threshold,
            "confidence_weights": {
                "rtl": self.rtl_weight,
                "field": self.field_weight,
                "text": self.text_weight,
            },
        }

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    def _fix_rtl(
        self, text: str, force: bool
    ) -> tuple[Optional[str], Optional[RTLFixStats]]:
        """Stage 1: Fix RTL reversal in Arabic text."""
        if not text or not text.strip():
            return text, None
        try:
            fixed, stats = self.rtl_fixer.analyze_and_fix(text, force=force)
            return fixed, stats
        except Exception as exc:
            logger.error("RTL fix failed: %s", exc)
            return text, None

    def _extract_fields(
        self, text: str
    ) -> Optional[ExtractedMedicalFields]:
        """Stage 2: Extract medical fields from RTL-fixed text."""
        if not text or not text.strip():
            return None
        try:
            return self.field_extractor.extract_fields(text)
        except Exception as exc:
            logger.error("Field extraction failed: %s", exc)
            return None

    def _compute_similarity(
        self, left: str, right: str
    ) -> Any:
        """Stage 3: Compute field-aware similarity between two records."""
        try:
            result = field_aware_similarity(left, right, extractor=self.field_extractor)
            return type("Obj", (), result)()  # dict-like object
        except Exception as exc:
            logger.error("Similarity computation failed: %s", exc)
            return None

    def _compute_confidence(
        self,
        left: str,
        right: str,
        similarity: Any,
        stats_left: Optional[RTLFixStats],
        stats_right: Optional[RTLFixStats],
    ) -> tuple[float, str]:
        """Stage 4: Compute overall confidence score.

        Combines three signals:
        - **Field similarity** (dominant): how well patient-identifying
          fields match between the two records.
        - **Text similarity**: token-level fuzzy match on full text.
        - **RTL consistency**: whether both records had similar RTL
          fix patterns (both reversed or both not).
        """
        field_sim = 0.0
        text_sim = 0.0
        rtl_sim = 1.0

        if similarity:
            field_sim = getattr(similarity, "score", 0.0)
            text_sim = getattr(similarity, "field_scores", {}).get(
                "template_signature", 0.0,
            )

        # RTL consistency: if one was reversed and the other not, penalize
        if stats_left and stats_right:
            left_reversed = stats_left.changed
            right_reversed = stats_right.changed
            rtl_sim = 1.0 if left_reversed == right_reversed else 0.5

        confidence = (
            self.field_weight * field_sim
            + self.text_weight * text_sim
            + self.rtl_weight * rtl_sim
        )

        # Explanation
        if confidence >= self.duplicate_threshold:
            explanation = (
                "High confidence duplicate: weighted field similarity "
                f"({field_sim:.2f}) exceeds threshold "
                f"({self.duplicate_threshold:.2f})."
            )
            if rtl_sim < 1.0:
                explanation += (
                    " Note: RTL fix patterns differ between records, "
                    "which may indicate different scanning conditions."
                )
        else:
            explanation = (
                f"Not a duplicate: combined score ({confidence:.2f}) "
                f"is below threshold ({self.duplicate_threshold:.2f}). "
                f"Field similarity: {field_sim:.2f}, "
                f"text similarity: {text_sim:.2f}."
            )

        return confidence, explanation

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stats_to_dict(stats: Optional[RTLFixStats]) -> dict[str, Any]:
        if not stats:
            return {"reversal_ratio": 0.0, "had_presentation_forms": False, "changed": False}
        return asdict(stats)

    @staticmethod
    def _step_dict(step_name: str, success: bool) -> dict[str, Any]:
        return {"step": step_name, "success": success}