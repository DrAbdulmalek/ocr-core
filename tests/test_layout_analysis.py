import math

import pytest

from ocr_core.layout_analysis import (
    AnalysisProvenance,
    LayoutAnalysisResult,
    ReconciliationAction,
    RegionHypothesis,
    reconcile,
)
from ocr_core.ui.models import LayoutDocument, MedicalRegionData, RegionType


def provenance():
    return AnalysisProvenance("test-analyzer", "1.0", "cfg-a", "image-1")


def hypothesis(region_id, x=1.0, y=2.0):
    return RegionHypothesis(region_id, RegionType.CLINICAL_SECTION, (x, y, 10.0, 8.0))


def document(*regions):
    return LayoutDocument("doc-1", (100, 100), regions=list(regions))


def test_analysis_result_is_deterministically_ordered_and_serializable():
    result = LayoutAnalysisResult.from_hypotheses(
        provenance(), [hypothesis("z"), hypothesis("a")]
    )
    assert [item.region_id for item in result.hypotheses] == ["a", "z"]
    assert LayoutAnalysisResult.from_dict(result.to_dict()) == result


def test_analysis_result_rejects_duplicate_region_ids():
    with pytest.raises(ValueError, match="duplicate region_id"):
        LayoutAnalysisResult.from_hypotheses(
            provenance(), [hypothesis("same"), hypothesis("same")]
        )


@pytest.mark.parametrize(
    "bbox",
    [(0, 0, math.nan, 1), (0, 0, math.inf, 1), (0, 0, -1, 1)],
)
def test_hypothesis_rejects_invalid_bbox(bbox):
    with pytest.raises(ValueError):
        RegionHypothesis("r1", RegionType.CLINICAL_SECTION, bbox)


def test_reconcile_does_not_mutate_document_and_protects_manual_region():
    region = MedicalRegionData(
        region_id="r1",
        bbox=(1, 2, 10, 8),
        text_verbatim="النص الأصلي",
        is_manually_edited=True,
    )
    before = region.to_dict()
    result = LayoutAnalysisResult.from_hypotheses(provenance(), [hypothesis("r1", 4, 5)])
    reconciliation = reconcile(document(region), result)

    assert reconciliation.has_conflicts
    assert reconciliation.items[0].action is ReconciliationAction.CONFLICT
    assert region.to_dict() == before
    assert region.text_verbatim == "النص الأصلي"


def test_reconcile_reports_addition_and_does_not_apply_it():
    doc = document()
    result = LayoutAnalysisResult.from_hypotheses(provenance(), [hypothesis("new")])
    reconciliation = reconcile(doc, result)

    assert reconciliation.items[0].action is ReconciliationAction.ADDED
    assert doc.regions == []


def test_reconcile_reports_non_manual_update_and_deletion():
    existing = MedicalRegionData(
        region_id="r1",
        bbox=(1, 2, 10, 8),
        is_manually_edited=False,
        source="layout_analyzer",
    )
    result = LayoutAnalysisResult.from_hypotheses(provenance(), [hypothesis("r1", 4, 5)])
    reconciliation = reconcile(document(existing), result)
    assert reconciliation.items[0].action is ReconciliationAction.UPDATE_PROPOSAL

    deletion = reconcile(document(existing), LayoutAnalysisResult(provenance(), ()))
    assert deletion.items[0].action is ReconciliationAction.DELETION_PROPOSAL


def test_missing_manual_region_is_a_conflict_not_a_deletion():
    existing = MedicalRegionData(
        region_id="r1",
        bbox=(1, 2, 10, 8),
        is_manually_edited=True,
    )
    reconciliation = reconcile(document(existing), LayoutAnalysisResult(provenance(), ()))
    assert reconciliation.items[0].action is ReconciliationAction.CONFLICT
    assert "manually edited" in reconciliation.items[0].reason


def test_reconcile_rejects_duplicate_document_region_ids():
    first = MedicalRegionData(region_id="same", bbox=(1, 2, 10, 8))
    second = MedicalRegionData(region_id="same", bbox=(4, 5, 10, 8))
    with pytest.raises(ValueError, match="duplicate region_id"):
        reconcile(document(first, second), LayoutAnalysisResult(provenance(), ()))


def test_analysis_provenance_preserves_explicit_run_identity():
    value = provenance()
    value = AnalysisProvenance(
        value.analyzer,
        value.analyzer_version,
        value.configuration_id,
        value.source_id,
        analysis_run_id="run-001",
    )
    restored = AnalysisProvenance.from_dict(value.to_dict())
    assert restored == value
    assert restored.analysis_run_id == "run-001"


def test_legacy_analysis_provenance_without_run_identity_remains_readable():
    restored = AnalysisProvenance.from_dict(
        {
            "analyzer": "test-analyzer",
            "analyzer_version": "1.0",
            "configuration_id": "cfg-a",
            "source_id": "image-1",
        }
    )
    assert restored.analysis_run_id is None


def test_analysis_provenance_rejects_blank_run_identity():
    with pytest.raises(ValueError, match="analysis_run_id"):
        AnalysisProvenance("test-analyzer", "1.0", "cfg-a", "image-1", analysis_run_id="  ")


def test_hypothesis_id_is_an_analysis_scoped_compatibility_alias():
    item = hypothesis("h-001")
    assert item.hypothesis_id == item.region_id
