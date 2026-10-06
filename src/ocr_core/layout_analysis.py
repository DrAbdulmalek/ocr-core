"""Minimal, non-destructive runtime contract for layout analysis.

This module deliberately contains no ML/layout engine. It turns analyzer
hypotheses into an explicit, reviewable reconciliation result without
mutating the canonical LayoutDocument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Iterable, Mapping

from .ui.models import LayoutDocument, MedicalRegionData, RegionType


def _validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    if len(bbox) != 4:
        raise ValueError("bbox must be (x, y, width, height)")
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError("bbox values must be finite")
    if bbox[2] < 0 or bbox[3] < 0:
        raise ValueError("bbox width and height must be non-negative")


@dataclass(frozen=True)
class AnalysisProvenance:
    """Identity of one layout-analysis event, separate from layout_version."""

    analyzer: str
    analyzer_version: str
    configuration_id: str
    source_id: str
    analysis_run_id: str | None = None

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.analyzer, self.analyzer_version, self.configuration_id, self.source_id)
        ):
            raise ValueError("analysis provenance fields are required")
        if self.analysis_run_id is not None and (
            not isinstance(self.analysis_run_id, str) or not self.analysis_run_id.strip()
        ):
            raise ValueError("analysis_run_id must be a non-empty string when provided")

    def to_dict(self) -> dict[str, str]:
        payload = {
            "analyzer": self.analyzer,
            "analyzer_version": self.analyzer_version,
            "configuration_id": self.configuration_id,
            "source_id": self.source_id,
        }
        if self.analysis_run_id is not None:
            payload["analysis_run_id"] = self.analysis_run_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalysisProvenance":
        return cls(
            analyzer=str(payload["analyzer"]),
            analyzer_version=str(payload["analyzer_version"]),
            configuration_id=str(payload["configuration_id"]),
            source_id=str(payload["source_id"]),
            analysis_run_id=(
                str(payload["analysis_run_id"])
                if payload.get("analysis_run_id") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class RegionHypothesis:
    """A proposed region; it is not persisted until an explicit consumer action."""

    region_id: str
    region_type: RegionType
    bbox: tuple[float, float, float, float]
    source: str = "layout_analyzer"

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("region_id is required")
        object.__setattr__(self, "region_type", RegionType(self.region_type))
        bbox = tuple(float(value) for value in self.bbox)
        _validate_bbox(bbox)  # type: ignore[arg-type]
        object.__setattr__(self, "bbox", bbox)

    @property
    def hypothesis_id(self) -> str:
        """Analysis-scoped compatibility alias for the existing hypothesis ID."""
        return self.region_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "region_type": self.region_type.value,
            "bbox": list(self.bbox),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegionHypothesis":
        return cls(
            region_id=str(payload["region_id"]),
            region_type=RegionType(str(payload["region_type"])),
            bbox=tuple(float(value) for value in payload["bbox"]),  # type: ignore[arg-type]
            source=str(payload.get("source", "layout_analyzer")),
        )


@dataclass(frozen=True)
class LayoutAnalysisResult:
    """Deterministic output of one analysis event."""

    provenance: AnalysisProvenance
    hypotheses: tuple[RegionHypothesis, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.hypotheses, key=lambda item: item.region_id))
        if ordered != self.hypotheses:
            object.__setattr__(self, "hypotheses", ordered)
        ids = [item.region_id for item in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("analysis result contains duplicate region_id values")

    @classmethod
    def from_hypotheses(
        cls, provenance: AnalysisProvenance, hypotheses: Iterable[RegionHypothesis]
    ) -> "LayoutAnalysisResult":
        return cls(provenance=provenance, hypotheses=tuple(hypotheses))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "hypotheses": [item.to_dict() for item in self.hypotheses],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LayoutAnalysisResult":
        return cls(
            provenance=AnalysisProvenance.from_dict(payload["provenance"]),
            hypotheses=tuple(
                RegionHypothesis.from_dict(item) for item in payload.get("hypotheses", [])
            ),
        )


class ReconciliationAction(str, Enum):
    UNCHANGED = "unchanged"
    ADDED = "added"
    UPDATE_PROPOSAL = "update_proposal"
    CONFLICT = "conflict"
    DELETION_PROPOSAL = "deletion_proposal"


@dataclass(frozen=True)
class ReconciliationItem:
    action: ReconciliationAction
    region_id: str
    existing: MedicalRegionData | None = None
    proposed: RegionHypothesis | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("region_id is required")


@dataclass(frozen=True)
class ReconciliationResult:
    """Reviewable diff. Applying it is intentionally outside this module."""

    analysis: LayoutAnalysisResult
    items: tuple[ReconciliationItem, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.items, key=lambda item: item.region_id))
        if ordered != self.items:
            object.__setattr__(self, "items", ordered)

    @property
    def has_conflicts(self) -> bool:
        return any(item.action is ReconciliationAction.CONFLICT for item in self.items)


def reconcile(
    document: LayoutDocument, analysis: LayoutAnalysisResult
) -> ReconciliationResult:
    """Compare analysis hypotheses with a document without mutating it.

    Matching is by persisted region_id only. Cross-run semantic matching is
    deliberately not inferred from geometry, type, or list position.
    """
    existing: dict[str, MedicalRegionData] = {}
    for region in document.regions:
        if region.region_id in existing:
            raise ValueError(f"document contains duplicate region_id: {region.region_id}")
        existing[region.region_id] = region

    proposed = {item.region_id: item for item in analysis.hypotheses}
    items: list[ReconciliationItem] = []

    for region_id, hypothesis in proposed.items():
        current = existing.get(region_id)
        if current is None:
            items.append(
                ReconciliationItem(
                    ReconciliationAction.ADDED,
                    region_id,
                    proposed=hypothesis,
                    reason="no persisted region with this identity",
                )
            )
            continue

        same = (
            current.region_type is hypothesis.region_type
            and current.bbox == hypothesis.bbox
        )
        if same:
            items.append(
                ReconciliationItem(
                    ReconciliationAction.UNCHANGED,
                    region_id,
                    existing=current,
                    proposed=hypothesis,
                )
            )
        elif current.is_manually_edited:
            items.append(
                ReconciliationItem(
                    ReconciliationAction.CONFLICT,
                    region_id,
                    existing=current,
                    proposed=hypothesis,
                    reason="manual edits have precedence",
                )
            )
        else:
            items.append(
                ReconciliationItem(
                    ReconciliationAction.UPDATE_PROPOSAL,
                    region_id,
                    existing=current,
                    proposed=hypothesis,
                    reason="automatic region may be reviewed for update",
                )
            )

    for region_id, current in existing.items():
        if region_id not in proposed:
            if current.is_manually_edited:
                items.append(
                    ReconciliationItem(
                        ReconciliationAction.CONFLICT,
                        region_id,
                        existing=current,
                        reason="analysis omitted a manually edited region",
                    )
                )
            else:
                items.append(
                    ReconciliationItem(
                        ReconciliationAction.DELETION_PROPOSAL,
                        region_id,
                        existing=current,
                        reason="analysis omitted an existing non-manual region",
                    )
                )

    return ReconciliationResult(analysis=analysis, items=tuple(items))
