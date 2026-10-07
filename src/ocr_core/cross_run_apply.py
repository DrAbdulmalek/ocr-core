"""Explicit, atomic application of reviewed cross-run reconciliation decisions.

Matching remains proposal-only. This module is the explicit mutation boundary:
it validates a complete decision set, builds a new LayoutDocument, and returns
an auditable apply result. The supplied document is never mutated in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from enum import Enum
from typing import Any, Mapping

from .cross_run_matching import (
    CrossRunMatchCandidate,
    CrossRunMatchState,
    CrossRunReconciliationResult,
)
from .layout_analysis import LayoutAnalysisResult
from .ui.models import LayoutDocument, MedicalRegionData


class ApplyAction(str, Enum):
    ACCEPT_MATCH = "accept_match"
    CREATE_NEW = "create_new"


@dataclass(frozen=True)
class ReconciliationApplyDecision:
    """One explicit mutation decision."""

    action: ApplyAction
    hypothesis_id: str
    persistent_region_id: str
    update_bbox: bool = False
    update_region_type: bool = False
    authorize_manual_override: bool = False
    authorize_resolution: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.hypothesis_id or not self.persistent_region_id:
            raise ValueError("hypothesis_id and persistent_region_id are required")
        if self.action is ApplyAction.CREATE_NEW and (
            self.update_bbox
            or self.update_region_type
            or self.authorize_manual_override
            or self.authorize_resolution
        ):
            raise ValueError("CREATE_NEW cannot use update/override/resolution flags")


@dataclass(frozen=True)
class AppliedReconciliationAction:
    action: ApplyAction
    hypothesis_id: str
    persistent_region_id: str
    changed_fields: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "hypothesis_id": self.hypothesis_id,
            "persistent_region_id": self.persistent_region_id,
            "changed_fields": list(self.changed_fields),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AppliedReconciliationAction":
        return cls(
            action=ApplyAction(str(payload["action"])),
            hypothesis_id=str(payload["hypothesis_id"]),
            persistent_region_id=str(payload["persistent_region_id"]),
            changed_fields=tuple(str(value) for value in payload.get("changed_fields", [])),
            reason=str(payload.get("reason", "")),
        )


@dataclass(frozen=True)
class ReconciliationApplyResult:
    """Auditable result of one successful atomic application."""

    document: LayoutDocument
    source_id: str
    analysis_run_id: str | None
    matching_policy_id: str
    expected_document_fingerprint: str
    applied: tuple[AppliedReconciliationAction, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "applied",
            tuple(
                sorted(
                    self.applied,
                    key=lambda item: (
                        item.persistent_region_id,
                        item.hypothesis_id,
                        item.action.value,
                    ),
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "source_id": self.source_id,
            "analysis_run_id": self.analysis_run_id,
            "matching_policy_id": self.matching_policy_id,
            "expected_document_fingerprint": self.expected_document_fingerprint,
            "applied": [item.to_dict() for item in self.applied],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReconciliationApplyResult":
        return cls(
            document=LayoutDocument.from_dict(payload["document"]),
            source_id=str(payload["source_id"]),
            analysis_run_id=(
                str(payload["analysis_run_id"])
                if payload.get("analysis_run_id") is not None
                else None
            ),
            matching_policy_id=str(payload["matching_policy_id"]),
            expected_document_fingerprint=str(payload["expected_document_fingerprint"]),
            applied=tuple(
                AppliedReconciliationAction.from_dict(item)
                for item in payload.get("applied", [])
            ),
        )


def document_fingerprint(document: LayoutDocument) -> str:
    """Return a deterministic fingerprint of the current document state."""

    payload = json.dumps(
        document.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_index(
    reconciliation: CrossRunReconciliationResult,
) -> dict[tuple[str, str], CrossRunMatchCandidate]:
    index: dict[tuple[str, str], CrossRunMatchCandidate] = {}
    for candidate in reconciliation.candidates:
        key = (candidate.hypothesis_id, candidate.persistent_region_id)
        if key in index:
            raise ValueError(f"duplicate reconciliation candidate: {key}")
        index[key] = candidate
    return index


def apply_cross_run_reconciliation(
    document: LayoutDocument,
    reconciliation: CrossRunReconciliationResult,
    decisions: tuple[ReconciliationApplyDecision, ...],
    *,
    expected_document_fingerprint: str,
) -> ReconciliationApplyResult:
    """Validate and atomically apply an explicit decision set.

    The input document is never mutated. Any validation failure raises before
    the returned document is constructed/committed.
    """

    actual_fingerprint = document_fingerprint(document)
    if expected_document_fingerprint != actual_fingerprint:
        raise ValueError("document state is stale; apply requires the expected fingerprint")

    existing = {region.region_id: region for region in document.regions}
    if len(existing) != len(document.regions):
        raise ValueError("document contains duplicate region_id values")

    hypotheses = {item.hypothesis_id: item for item in reconciliation.analysis.hypotheses}
    if len(hypotheses) != len(reconciliation.analysis.hypotheses):
        raise ValueError("analysis contains duplicate hypothesis IDs")

    candidates = _candidate_index(reconciliation)
    seen_decisions: set[tuple[str, str]] = set()

    for decision in decisions:
        key = (decision.hypothesis_id, decision.persistent_region_id)
        if key in seen_decisions:
            raise ValueError(f"duplicate apply decision: {key}")
        seen_decisions.add(key)

        hypothesis = hypotheses.get(decision.hypothesis_id)
        if hypothesis is None:
            raise ValueError("apply decision references an unknown hypothesis")

        if decision.action is ApplyAction.CREATE_NEW:
            if decision.hypothesis_id not in reconciliation.unmatched_new:
                raise ValueError("CREATE_NEW requires an UNMATCHED_NEW hypothesis")
            if decision.persistent_region_id in existing:
                raise ValueError("CREATE_NEW persistent_region_id already exists")
            if key in candidates:
                raise ValueError("CREATE_NEW cannot target an existing reconciliation candidate")
            continue

        candidate = candidates.get(key)
        if candidate is None:
            raise ValueError("apply decision references a candidate not present in reconciliation")

        if candidate.state not in {
            CrossRunMatchState.MATCHED,
            CrossRunMatchState.CONFLICT,
            CrossRunMatchState.AMBIGUOUS,
        }:
            raise ValueError("ACCEPT_MATCH requires a resolvable candidate")

        if candidate.state is not CrossRunMatchState.MATCHED and not decision.authorize_resolution:
            raise ValueError("non-MATCHED candidate requires explicit resolution authorization")

        target = existing.get(decision.persistent_region_id)
        if target is None:
            raise ValueError("target persistent region does not exist")

        if target.is_manually_edited and not decision.authorize_manual_override:
            raise ValueError("manual region requires explicit protected-region authorization")

        if (candidate.state is CrossRunMatchState.AMBIGUOUS or candidate.state is CrossRunMatchState.CONFLICT) and not decision.reason.strip():
            raise ValueError("resolved candidate requires an explicit reason")

    new_regions = [
        MedicalRegionData.from_dict(region.to_dict()) for region in document.regions
    ]
    by_id = {region.region_id: region for region in new_regions}
    applied: list[AppliedReconciliationAction] = []

    for decision in decisions:
        hypothesis = hypotheses[decision.hypothesis_id]

        if decision.action is ApplyAction.CREATE_NEW:
            new_region = MedicalRegionData(
                region_id=decision.persistent_region_id,
                region_type=hypothesis.region_type,
                bbox=hypothesis.bbox,
                source=hypothesis.source,
                is_manually_edited=False,
            )
            new_regions.append(new_region)
            by_id[new_region.region_id] = new_region
            applied.append(
                AppliedReconciliationAction(
                    action=decision.action,
                    hypothesis_id=decision.hypothesis_id,
                    persistent_region_id=decision.persistent_region_id,
                    changed_fields=("region_created",),
                    reason=decision.reason,
                )
            )
            continue

        current = by_id[decision.persistent_region_id]
        changed: list[str] = []
        if decision.update_bbox:
            current.bbox = hypothesis.bbox
            changed.append("bbox")
        if decision.update_region_type:
            current.region_type = hypothesis.region_type
            changed.append("region_type")
        if decision.authorize_manual_override and not current.is_manually_edited:
            current.is_manually_edited = True
            changed.append("is_manually_edited")

        applied.append(
            AppliedReconciliationAction(
                action=decision.action,
                hypothesis_id=decision.hypothesis_id,
                persistent_region_id=decision.persistent_region_id,
                changed_fields=tuple(changed),
                reason=decision.reason,
            )
        )

    result_document = LayoutDocument(
        document_id=document.document_id,
        image_dimensions=document.image_dimensions,
        dpi=document.dpi,
        layout_version=document.layout_version,
        regions=new_regions,
    )

    return ReconciliationApplyResult(
        document=result_document,
        source_id=reconciliation.analysis.provenance.source_id,
        analysis_run_id=reconciliation.analysis.provenance.analysis_run_id,
        matching_policy_id=reconciliation.config.policy_id,
        expected_document_fingerprint=expected_document_fingerprint,
        applied=tuple(applied),
    )
