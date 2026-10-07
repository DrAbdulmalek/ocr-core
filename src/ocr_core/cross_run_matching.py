"""Explicit cross-run matching runtime.

This module proposes matches between a new analysis result and persistent
regions without mutating LayoutDocument. It intentionally uses only
documented, deterministic evidence; hypothesis IDs are never matching keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .layout_analysis import LayoutAnalysisResult
from .ui.models import LayoutDocument, MedicalRegionData


class CrossRunMatchState(str, Enum):
    MATCHED = "matched"
    UNMATCHED_NEW = "unmatched_new"
    UNMATCHED_EXISTING = "unmatched_existing"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class CrossRunMatchingConfig:
    """Versioned policy for the deterministic baseline matcher."""

    policy_id: str = "geometry-iou-v1"
    min_iou: float = 0.5

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id is required")
        if not 0.0 <= self.min_iou <= 1.0:
            raise ValueError("min_iou must be between 0 and 1")


@dataclass(frozen=True)
class MatchEvidence:
    """Auditable evidence for one hypothesis/persistent-region candidate."""

    geometry_iou: float
    region_type_equal: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.geometry_iou <= 1.0:
            raise ValueError("geometry_iou must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry_iou": self.geometry_iou,
            "region_type_equal": self.region_type_equal,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MatchEvidence":
        return cls(
            geometry_iou=float(payload["geometry_iou"]),
            region_type_equal=bool(payload["region_type_equal"]),
        )


@dataclass(frozen=True)
class CrossRunMatchCandidate:
    """One explicit hypothesis -> persistent-region proposal."""

    hypothesis_id: str
    persistent_region_id: str
    evidence: MatchEvidence
    state: CrossRunMatchState = CrossRunMatchState.AMBIGUOUS
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.hypothesis_id or not self.persistent_region_id:
            raise ValueError("hypothesis_id and persistent_region_id are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "persistent_region_id": self.persistent_region_id,
            "evidence": self.evidence.to_dict(),
            "state": self.state.value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CrossRunMatchCandidate":
        return cls(
            hypothesis_id=str(payload["hypothesis_id"]),
            persistent_region_id=str(payload["persistent_region_id"]),
            evidence=MatchEvidence.from_dict(payload["evidence"]),
            state=CrossRunMatchState(str(payload["state"])),
            reason=str(payload.get("reason", "")),
        )


@dataclass(frozen=True)
class CrossRunReconciliationResult:
    """Immutable, reviewable cross-run proposal; never mutates the document."""

    analysis: LayoutAnalysisResult
    config: CrossRunMatchingConfig
    candidates: tuple[CrossRunMatchCandidate, ...] = field(default_factory=tuple)
    unmatched_new: tuple[str, ...] = field(default_factory=tuple)
    unmatched_existing: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(
                self.candidates,
                key=lambda item: (
                    item.hypothesis_id,
                    item.persistent_region_id,
                    item.state.value,
                ),
            )
        )
        if ordered != self.candidates:
            object.__setattr__(self, "candidates", ordered)
        object.__setattr__(self, "unmatched_new", tuple(sorted(set(self.unmatched_new))))
        object.__setattr__(
            self, "unmatched_existing", tuple(sorted(set(self.unmatched_existing)))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis.to_dict(),
            "config": {
                "policy_id": self.config.policy_id,
                "min_iou": self.config.min_iou,
            },
            "candidates": [item.to_dict() for item in self.candidates],
            "unmatched_new": list(self.unmatched_new),
            "unmatched_existing": list(self.unmatched_existing),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CrossRunReconciliationResult":
        config_payload = payload["config"]
        return cls(
            analysis=LayoutAnalysisResult.from_dict(payload["analysis"]),
            config=CrossRunMatchingConfig(
                policy_id=str(config_payload["policy_id"]),
                min_iou=float(config_payload["min_iou"]),
            ),
            candidates=tuple(
                CrossRunMatchCandidate.from_dict(item)
                for item in payload.get("candidates", [])
            ),
            unmatched_new=tuple(str(value) for value in payload.get("unmatched_new", [])),
            unmatched_existing=tuple(
                str(value) for value in payload.get("unmatched_existing", [])
            ),
        )


def _intersection_over_union(left, right) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    left_edge = max(lx, rx)
    top_edge = max(ly, ry)
    right_edge = min(lx + lw, rx + rw)
    bottom_edge = min(ly + lh, ry + rh)
    intersection = max(0.0, right_edge - left_edge) * max(
        0.0, bottom_edge - top_edge
    )
    union = (lw * lh) + (rw * rh) - intersection
    return 0.0 if union == 0.0 else intersection / union


def match_cross_run(
    document: LayoutDocument,
    analysis: LayoutAnalysisResult,
    *,
    source_id: str,
    config: CrossRunMatchingConfig | None = None,
) -> CrossRunReconciliationResult:
    """Generate deterministic, non-destructive cross-run match proposals."""

    config = config or CrossRunMatchingConfig()
    if not source_id.strip():
        raise ValueError("source_id is required")
    if analysis.provenance.source_id != source_id:
        raise ValueError("analysis source_id does not match target source_id")

    existing = {}
    for region in document.regions:
        if region.region_id in existing:
            raise ValueError(f"document contains duplicate region_id: {region.region_id}")
        existing[region.region_id] = region

    by_hypothesis = {}
    by_region = {}

    for hypothesis in analysis.hypotheses:
        for region in existing.values():
            iou = _intersection_over_union(hypothesis.bbox, region.bbox)
            if iou < config.min_iou:
                continue
            evidence = MatchEvidence(
                geometry_iou=iou,
                region_type_equal=hypothesis.region_type is region.region_type,
            )
            candidate = CrossRunMatchCandidate(
                hypothesis_id=hypothesis.hypothesis_id,
                persistent_region_id=region.region_id,
                evidence=evidence,
            )
            by_hypothesis.setdefault(hypothesis.hypothesis_id, []).append(candidate)
            by_region.setdefault(region.region_id, []).append(candidate)

    candidates = []
    for hypothesis in analysis.hypotheses:
        proposals = by_hypothesis.get(hypothesis.hypothesis_id, [])
        if not proposals:
            continue
        if len(proposals) != 1:
            candidates.extend(
                CrossRunMatchCandidate(
                    item.hypothesis_id,
                    item.persistent_region_id,
                    item.evidence,
                    CrossRunMatchState.AMBIGUOUS,
                    "multiple persistent-region candidates",
                )
                for item in proposals
            )
            continue

        item = proposals[0]
        competing = by_region[item.persistent_region_id]
        if len(competing) != 1:
            candidates.append(
                CrossRunMatchCandidate(
                    item.hypothesis_id,
                    item.persistent_region_id,
                    item.evidence,
                    CrossRunMatchState.AMBIGUOUS,
                    "persistent region has competing hypotheses",
                )
            )
            continue

        region = existing[item.persistent_region_id]
        if region.is_manually_edited and (
            not item.evidence.region_type_equal or item.evidence.geometry_iou < 1.0
        ):
            candidates.append(
                CrossRunMatchCandidate(
                    item.hypothesis_id,
                    item.persistent_region_id,
                    item.evidence,
                    CrossRunMatchState.CONFLICT,
                    "manual region geometry/type has precedence",
                )
            )
        else:
            candidates.append(
                CrossRunMatchCandidate(
                    item.hypothesis_id,
                    item.persistent_region_id,
                    item.evidence,
                    CrossRunMatchState.MATCHED,
                    "unique explicit evidence candidate",
                )
            )

    candidate_hypotheses = {item.hypothesis_id for item in candidates}
    candidate_regions = {item.persistent_region_id for item in candidates}
    return CrossRunReconciliationResult(
        analysis=analysis,
        config=config,
        candidates=tuple(candidates),
        unmatched_new=tuple(
            item.hypothesis_id
            for item in analysis.hypotheses
            if item.hypothesis_id not in candidate_hypotheses
        ),
        unmatched_existing=tuple(
            region.region_id
            for region in existing.values()
            if region.region_id not in candidate_regions
        ),
    )
