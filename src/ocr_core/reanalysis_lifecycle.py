"""Explicit, non-mutating lifecycle records for layout re-analysis.

This module records lifecycle/provenance events without changing LayoutDocument.
Matching and apply remain separate mutation/proposal layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

from .cross_run_apply import document_fingerprint
from .ui.models import LayoutDocument


class ReanalysisLifecycleState(str, Enum):
    INFERRED = "inferred"
    REVIEWED = "reviewed"
    APPLIED = "applied"
    MANUAL = "manual"
    CONFLICTED = "conflicted"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class ReanalysisLifecycleRecord:
    """Immutable audit record; constructing it never mutates a document."""

    state: ReanalysisLifecycleState
    document_id: str
    source_id: str
    analysis_run_id: str | None = None
    hypothesis_id: str | None = None
    persistent_region_id: str | None = None
    expected_document_fingerprint: str | None = None
    policy_id: str | None = None
    changed_fields: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""
    supersedes_analysis_run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.document_id or not self.source_id:
            raise ValueError("document_id and source_id are required")
        if self.state in {
            ReanalysisLifecycleState.INFERRED,
            ReanalysisLifecycleState.REVIEWED,
            ReanalysisLifecycleState.CONFLICTED,
        } and not self.hypothesis_id:
            raise ValueError(f"{self.state.value} records require hypothesis_id")
        if self.state in {
            ReanalysisLifecycleState.APPLIED,
            ReanalysisLifecycleState.MANUAL,
        } and not self.persistent_region_id:
            raise ValueError(f"{self.state.value} records require persistent_region_id")
        if self.state is ReanalysisLifecycleState.SUPERSEDED and not self.supersedes_analysis_run_id:
            raise ValueError("superseded records require supersedes_analysis_run_id")
        if self.expected_document_fingerprint is not None and len(self.expected_document_fingerprint) != 64:
            raise ValueError("expected_document_fingerprint must be a SHA-256 hex digest")
        object.__setattr__(self, "changed_fields", tuple(sorted(set(self.changed_fields))))

    @classmethod
    def from_document(
        cls,
        document: LayoutDocument,
        *,
        state: ReanalysisLifecycleState,
        source_id: str,
        analysis_run_id: str | None = None,
        hypothesis_id: str | None = None,
        persistent_region_id: str | None = None,
        policy_id: str | None = None,
        changed_fields: tuple[str, ...] = (),
        reason: str = "",
        supersedes_analysis_run_id: str | None = None,
    ) -> "ReanalysisLifecycleRecord":
        return cls(
            state=state,
            document_id=document.document_id,
            source_id=source_id,
            analysis_run_id=analysis_run_id,
            hypothesis_id=hypothesis_id,
            persistent_region_id=persistent_region_id,
            expected_document_fingerprint=document_fingerprint(document),
            policy_id=policy_id,
            changed_fields=changed_fields,
            reason=reason,
            supersedes_analysis_run_id=supersedes_analysis_run_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "document_id": self.document_id,
            "source_id": self.source_id,
            "analysis_run_id": self.analysis_run_id,
            "hypothesis_id": self.hypothesis_id,
            "persistent_region_id": self.persistent_region_id,
            "expected_document_fingerprint": self.expected_document_fingerprint,
            "policy_id": self.policy_id,
            "changed_fields": list(self.changed_fields),
            "reason": self.reason,
            "supersedes_analysis_run_id": self.supersedes_analysis_run_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReanalysisLifecycleRecord":
        return cls(
            state=ReanalysisLifecycleState(str(payload["state"])),
            document_id=str(payload["document_id"]),
            source_id=str(payload["source_id"]),
            analysis_run_id=(str(payload["analysis_run_id"]) if payload.get("analysis_run_id") is not None else None),
            hypothesis_id=(str(payload["hypothesis_id"]) if payload.get("hypothesis_id") is not None else None),
            persistent_region_id=(str(payload["persistent_region_id"]) if payload.get("persistent_region_id") is not None else None),
            expected_document_fingerprint=(str(payload["expected_document_fingerprint"]) if payload.get("expected_document_fingerprint") is not None else None),
            policy_id=(str(payload["policy_id"]) if payload.get("policy_id") is not None else None),
            changed_fields=tuple(str(value) for value in payload.get("changed_fields", [])),
            reason=str(payload.get("reason", "")),
            supersedes_analysis_run_id=(str(payload["supersedes_analysis_run_id"]) if payload.get("supersedes_analysis_run_id") is not None else None),
        )


def validate_lifecycle_record(
    record: ReanalysisLifecycleRecord,
    document: LayoutDocument,
    *,
    source_id: str,
) -> None:
    """Validate that a record belongs to the supplied current document state."""
    if record.document_id != document.document_id:
        raise ValueError("lifecycle record targets a different document")
    if record.source_id != source_id:
        raise ValueError("lifecycle record source_id does not match")
    if record.expected_document_fingerprint is not None:
        actual = document_fingerprint(document)
        if record.expected_document_fingerprint != actual:
            raise ValueError("lifecycle record targets a stale document state")


def supersede_lifecycle_record(
    record: ReanalysisLifecycleRecord,
    *,
    superseding_analysis_run_id: str,
) -> ReanalysisLifecycleRecord:
    """Create a historical supersession record without mutating the original."""
    if not superseding_analysis_run_id:
        raise ValueError("superseding_analysis_run_id is required")
    return ReanalysisLifecycleRecord(
        state=ReanalysisLifecycleState.SUPERSEDED,
        document_id=record.document_id,
        source_id=record.source_id,
        analysis_run_id=superseding_analysis_run_id,
        hypothesis_id=record.hypothesis_id,
        persistent_region_id=record.persistent_region_id,
        expected_document_fingerprint=record.expected_document_fingerprint,
        policy_id=record.policy_id,
        changed_fields=record.changed_fields,
        reason=record.reason,
        supersedes_analysis_run_id=record.analysis_run_id or superseding_analysis_run_id,
    )


def lifecycle_record_fingerprint(record: ReanalysisLifecycleRecord) -> str:
    payload = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
