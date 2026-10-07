from copy import deepcopy

import pytest

from ocr_core.reanalysis_lifecycle import (
    ReanalysisLifecycleRecord,
    ReanalysisLifecycleState,
    lifecycle_record_fingerprint,
    supersede_lifecycle_record,
    validate_lifecycle_record,
)
from ocr_core.ui.models import LayoutDocument, MedicalRegionData, RegionType


def make_document() -> LayoutDocument:
    return LayoutDocument(
        document_id="doc-1",
        image_dimensions=(1000, 800),
        regions=[
            MedicalRegionData(
                region_id="persistent-1",
                region_type=RegionType.CLINICAL_SECTION,
                bbox=(10, 20, 100, 50),
                source="analyzer",
                is_manually_edited=False,
            ),
            MedicalRegionData(
                region_id="manual-1",
                region_type=RegionType.FOOTER_SIGNATURE,
                bbox=(20, 700, 200, 40),
                source="user",
                is_manually_edited=True,
            ),
        ],
    )


def test_lifecycle_record_creation_does_not_mutate_document():
    document = make_document()
    before = deepcopy(document.to_dict())
    record = ReanalysisLifecycleRecord.from_document(
        document,
        state=ReanalysisLifecycleState.INFERRED,
        source_id="doc-1",
        analysis_run_id="run-1",
        hypothesis_id="hyp-1",
    )
    assert record.persistent_region_id is None
    assert document.to_dict() == before


def test_persistent_identity_is_explicit_and_not_hypothesis_identity():
    record = ReanalysisLifecycleRecord(
        state=ReanalysisLifecycleState.APPLIED,
        document_id="doc-1",
        source_id="doc-1",
        analysis_run_id="run-2",
        hypothesis_id="hyp-new",
        persistent_region_id="persistent-1",
    )
    assert record.hypothesis_id == "hyp-new"
    assert record.persistent_region_id == "persistent-1"
    assert record.hypothesis_id != record.persistent_region_id


def test_manual_state_is_audit_state_not_implicit_machine_state():
    record = ReanalysisLifecycleRecord(
        state=ReanalysisLifecycleState.APPLIED,
        document_id="doc-1",
        source_id="doc-1",
        analysis_run_id="run-2",
        hypothesis_id="hyp-1",
        persistent_region_id="manual-1",
    )
    assert record.state is ReanalysisLifecycleState.APPLIED
    assert record.persistent_region_id == "manual-1"


def test_omission_has_no_deletion_semantics():
    record = ReanalysisLifecycleRecord(
        state=ReanalysisLifecycleState.SUPERSEDED,
        document_id="doc-1",
        source_id="doc-1",
        analysis_run_id="run-2",
        persistent_region_id="persistent-1",
        supersedes_analysis_run_id="run-1",
    )
    assert record.state is ReanalysisLifecycleState.SUPERSEDED
    assert record.persistent_region_id == "persistent-1"


def test_overlap_or_conflict_remains_explicit():
    record = ReanalysisLifecycleRecord(
        state=ReanalysisLifecycleState.CONFLICTED,
        document_id="doc-1",
        source_id="doc-1",
        analysis_run_id="run-3",
        hypothesis_id="hyp-3",
        persistent_region_id="persistent-1",
        reason="overlapping hypotheses require review",
    )
    assert record.state is ReanalysisLifecycleState.CONFLICTED


def test_stale_document_is_rejected():
    document = make_document()
    record = ReanalysisLifecycleRecord.from_document(
        document,
        state=ReanalysisLifecycleState.APPLIED,
        source_id="doc-1",
        analysis_run_id="run-1",
        hypothesis_id="hyp-1",
        persistent_region_id="persistent-1",
    )
    document.regions[0].bbox = (11, 20, 100, 50)
    with pytest.raises(ValueError, match="stale"):
        validate_lifecycle_record(record, document, source_id="doc-1")


def test_source_mismatch_is_rejected():
    document = make_document()
    record = ReanalysisLifecycleRecord.from_document(
        document,
        state=ReanalysisLifecycleState.INFERRED,
        source_id="source-a",
        hypothesis_id="hyp-1",
    )
    with pytest.raises(ValueError, match="source_id"):
        validate_lifecycle_record(record, document, source_id="source-b")


def test_superseding_preserves_history_without_mutation():
    record = ReanalysisLifecycleRecord(
        state=ReanalysisLifecycleState.INFERRED,
        document_id="doc-1",
        source_id="source-a",
        analysis_run_id="run-1",
        hypothesis_id="hyp-1",
    )
    superseded = supersede_lifecycle_record(record, superseding_analysis_run_id="run-2")
    assert record.state is ReanalysisLifecycleState.INFERRED
    assert record.analysis_run_id == "run-1"
    assert superseded.state is ReanalysisLifecycleState.SUPERSEDED
    assert superseded.analysis_run_id == "run-2"
    assert superseded.supersedes_analysis_run_id == "run-1"


def test_serialization_is_deterministic_and_round_trips():
    record = ReanalysisLifecycleRecord(
        state=ReanalysisLifecycleState.APPLIED,
        document_id="doc-1",
        source_id="source-a",
        analysis_run_id="run-1",
        hypothesis_id="hyp-1",
        persistent_region_id="persistent-1",
        policy_id="policy-v1",
        changed_fields=("region_type", "bbox", "bbox"),
        reason="approved",
    )
    payload = record.to_dict()
    restored = ReanalysisLifecycleRecord.from_dict(payload)
    assert restored == record
    assert lifecycle_record_fingerprint(record) == lifecycle_record_fingerprint(restored)


def test_legacy_payload_without_optional_fields_is_readable():
    payload = {
        "state": "inferred",
        "document_id": "doc-1",
        "source_id": "source-a",
        "hypothesis_id": "hyp-1",
    }
    restored = ReanalysisLifecycleRecord.from_dict(payload)
    assert restored.analysis_run_id is None
    assert restored.persistent_region_id is None


def test_invalid_state_requirements_are_rejected():
    with pytest.raises(ValueError):
        ReanalysisLifecycleRecord(
            state=ReanalysisLifecycleState.INFERRED,
            document_id="doc-1",
            source_id="source-a",
        )
    with pytest.raises(ValueError):
        ReanalysisLifecycleRecord(
            state=ReanalysisLifecycleState.APPLIED,
            document_id="doc-1",
            source_id="source-a",
        )


def test_unrelated_regions_are_not_mutated():
    document = make_document()
    before = document.to_json()
    _ = ReanalysisLifecycleRecord.from_document(
        document,
        state=ReanalysisLifecycleState.REVIEWED,
        source_id="source-a",
        analysis_run_id="run-4",
        hypothesis_id="hyp-4",
    )
    assert document.to_json() == before
