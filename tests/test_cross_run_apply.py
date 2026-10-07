from __future__ import annotations

import copy

import pytest

from ocr_core.cross_run_apply import (
    ApplyAction,
    ReconciliationApplyDecision,
    apply_cross_run_reconciliation,
    document_fingerprint,
)
from ocr_core.cross_run_matching import (
    CrossRunMatchCandidate,
    CrossRunMatchState,
    CrossRunMatchingConfig,
    CrossRunReconciliationResult,
    MatchEvidence,
)
from ocr_core.layout_analysis import (
    AnalysisProvenance,
    LayoutAnalysisResult,
    RegionHypothesis,
)
from ocr_core.ui.models import LayoutDocument, MedicalRegionData, RegionType


def _analysis(*hypotheses: RegionHypothesis) -> LayoutAnalysisResult:
    return LayoutAnalysisResult.from_hypotheses(
        AnalysisProvenance(
            analyzer="test",
            analyzer_version="1",
            configuration_id="cfg",
            source_id="doc-1",
            analysis_run_id="run-1",
        ),
        hypotheses,
    )


def _candidate(
    hypothesis_id: str,
    region_id: str,
    state: CrossRunMatchState,
    iou: float = 1.0,
    same_type: bool = True,
) -> CrossRunMatchCandidate:
    return CrossRunMatchCandidate(
        hypothesis_id=hypothesis_id,
        persistent_region_id=region_id,
        evidence=MatchEvidence(iou, same_type),
        state=state,
    )


def _document(*regions: MedicalRegionData) -> LayoutDocument:
    return LayoutDocument("doc-1", (100, 100), regions=list(regions))


def test_apply_updates_only_explicitly_authorized_fields_and_preserves_id():
    current = MedicalRegionData(
        region_id="persistent-1",
        region_type=RegionType.CLINICAL_SECTION,
        bbox=(1, 2, 10, 10),
        source="baseline",
        is_manually_edited=False,
    )
    hypothesis = RegionHypothesis(
        region_id="hyp-1",
        region_type=RegionType.MEDICAL_TABLE,
        bbox=(3, 4, 20, 20),
    )
    analysis = _analysis(hypothesis)
    reconciliation = CrossRunReconciliationResult(
        analysis=analysis,
        config=CrossRunMatchingConfig(),
        candidates=(_candidate("hyp-1", "persistent-1", CrossRunMatchState.MATCHED),),
    )
    document = _document(current)
    original = copy.deepcopy(document)

    result = apply_cross_run_reconciliation(
        document,
        reconciliation,
        (
            ReconciliationApplyDecision(
                ApplyAction.ACCEPT_MATCH,
                "hyp-1",
                "persistent-1",
                update_bbox=True,
                update_region_type=True,
            ),
        ),
        expected_document_fingerprint=document_fingerprint(document),
    )

    assert document.to_dict() == original.to_dict()
    updated = result.document.regions[0]
    assert updated.region_id == "persistent-1"
    assert updated.bbox == (3, 4, 20, 20)
    assert updated.region_type is RegionType.MEDICAL_TABLE
    assert result.applied[0].changed_fields == ("bbox", "region_type")


def test_manual_region_requires_explicit_override():
    region = MedicalRegionData(
        region_id="persistent-1",
        bbox=(1, 2, 10, 10),
        is_manually_edited=True,
    )
    hypothesis = RegionHypothesis("hyp-1", RegionType.CLINICAL_SECTION, (1, 2, 10, 10))
    reconciliation = CrossRunReconciliationResult(
        _analysis(hypothesis),
        CrossRunMatchingConfig(),
        (_candidate("hyp-1", "persistent-1", CrossRunMatchState.MATCHED),),
    )
    document = _document(region)

    with pytest.raises(ValueError, match="manual region"):
        apply_cross_run_reconciliation(
            document,
            reconciliation,
            (ReconciliationApplyDecision(
                ApplyAction.ACCEPT_MATCH, "hyp-1", "persistent-1", update_bbox=True
            ),),
            expected_document_fingerprint=document_fingerprint(document),
        )


def test_ambiguous_candidate_requires_explicit_resolution():
    region = MedicalRegionData(region_id="persistent-1", bbox=(0, 0, 10, 10))
    hypothesis = RegionHypothesis("hyp-1", RegionType.CLINICAL_SECTION, (0, 0, 10, 10))
    reconciliation = CrossRunReconciliationResult(
        _analysis(hypothesis),
        CrossRunMatchingConfig(),
        (_candidate("hyp-1", "persistent-1", CrossRunMatchState.AMBIGUOUS),),
    )
    document = _document(region)

    with pytest.raises(ValueError, match="resolution"):
        apply_cross_run_reconciliation(
            document,
            reconciliation,
            (ReconciliationApplyDecision(
                ApplyAction.ACCEPT_MATCH, "hyp-1", "persistent-1"
            ),),
            expected_document_fingerprint=document_fingerprint(document),
        )


def test_unmatched_new_requires_explicit_create_and_never_uses_hypothesis_id():
    hypothesis = RegionHypothesis("hyp-new", RegionType.MEDICAL_TABLE, (5, 5, 20, 10))
    reconciliation = CrossRunReconciliationResult(
        _analysis(hypothesis),
        CrossRunMatchingConfig(),
        candidates=(),
        unmatched_new=("hyp-new",),
    )
    document = _document()

    with pytest.raises(ValueError, match="UNMATCHED_NEW"):
        apply_cross_run_reconciliation(
            document,
            reconciliation,
            (ReconciliationApplyDecision(
                ApplyAction.ACCEPT_MATCH, "hyp-new", "persistent-new"
            ),),
            expected_document_fingerprint=document_fingerprint(document),
        )

    result = apply_cross_run_reconciliation(
        document,
        reconciliation,
        (ReconciliationApplyDecision(
            ApplyAction.CREATE_NEW, "hyp-new", "persistent-new"
        ),),
        expected_document_fingerprint=document_fingerprint(document),
    )
    assert result.document.regions[0].region_id == "persistent-new"
    assert result.document.regions[0].region_id != "hyp-new"


def test_unmatched_existing_is_not_deleted_without_decision():
    region = MedicalRegionData(region_id="persistent-1", bbox=(1, 1, 5, 5))
    reconciliation = CrossRunReconciliationResult(
        _analysis(),
        CrossRunMatchingConfig(),
        candidates=(),
        unmatched_existing=("persistent-1",),
    )
    document = _document(region)

    result = apply_cross_run_reconciliation(
        document,
        reconciliation,
        (),
        expected_document_fingerprint=document_fingerprint(document),
    )
    assert [r.region_id for r in result.document.regions] == ["persistent-1"]


def test_validation_failure_is_atomic():
    region = MedicalRegionData(region_id="persistent-1", bbox=(1, 1, 5, 5))
    hypothesis = RegionHypothesis("hyp-1", RegionType.MEDICAL_TABLE, (2, 2, 8, 8))
    reconciliation = CrossRunReconciliationResult(
        _analysis(hypothesis),
        CrossRunMatchingConfig(),
        (_candidate("hyp-1", "persistent-1", CrossRunMatchState.MATCHED),),
    )
    document = _document(region)
    before = document.to_dict()

    with pytest.raises(ValueError, match="manual region"):
        apply_cross_run_reconciliation(
            document,
            reconciliation,
            (
                ReconciliationApplyDecision(
                    ApplyAction.ACCEPT_MATCH,
                    "hyp-1",
                    "persistent-1",
                    update_bbox=True,
                ),
            ),
            expected_document_fingerprint=document_fingerprint(document),
        )

    assert document.to_dict() == before


def test_stale_document_is_rejected():
    region = MedicalRegionData(region_id="persistent-1", bbox=(1, 1, 5, 5), is_manually_edited=False)
    hypothesis = RegionHypothesis("hyp-1", RegionType.CLINICAL_SECTION, (1, 1, 5, 5))
    reconciliation = CrossRunReconciliationResult(
        _analysis(hypothesis),
        CrossRunMatchingConfig(),
        (_candidate("hyp-1", "persistent-1", CrossRunMatchState.MATCHED),),
    )
    document = _document(region)

    with pytest.raises(ValueError, match="stale"):
        apply_cross_run_reconciliation(
            document,
            reconciliation,
            (),
            expected_document_fingerprint="0" * 64,
        )


def test_apply_result_is_serializable_and_deterministic():
    region = MedicalRegionData(region_id="persistent-1", bbox=(1, 1, 5, 5), is_manually_edited=False)
    hypothesis = RegionHypothesis("hyp-1", RegionType.CLINICAL_SECTION, (1, 1, 5, 5))
    reconciliation = CrossRunReconciliationResult(
        _analysis(hypothesis),
        CrossRunMatchingConfig(),
        (_candidate("hyp-1", "persistent-1", CrossRunMatchState.MATCHED),),
    )
    document = _document(region)
    result = apply_cross_run_reconciliation(
        document,
        reconciliation,
        (ReconciliationApplyDecision(
            ApplyAction.ACCEPT_MATCH, "hyp-1", "persistent-1"
        ),),
        expected_document_fingerprint=document_fingerprint(document),
    )
    payload = result.to_dict()
    assert payload["source_id"] == "doc-1"
    assert payload["analysis_run_id"] == "run-1"
    assert payload["matching_policy_id"] == "geometry-iou-v1"
    assert payload["applied"][0]["persistent_region_id"] == "persistent-1"
