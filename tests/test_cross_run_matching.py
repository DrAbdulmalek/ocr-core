from ocr_core.cross_run_matching import (
    CrossRunMatchState,
    CrossRunMatchingConfig,
    CrossRunReconciliationResult,
    match_cross_run,
)
from ocr_core.layout_analysis import AnalysisProvenance, LayoutAnalysisResult, RegionHypothesis
from ocr_core.ui.models import LayoutDocument, MedicalRegionData, RegionType


def analysis(*hypotheses, source_id="doc-1", run_id="run-2"):
    return LayoutAnalysisResult.from_hypotheses(
        AnalysisProvenance("baseline", "1", "cfg", source_id, run_id),
        hypotheses,
    )


def hypothesis(hid, bbox=(10, 10, 20, 20), region_type=RegionType.CLINICAL_SECTION):
    return RegionHypothesis(hid, region_type, bbox)


def document(*regions):
    return LayoutDocument("doc-1", (100, 100), regions=list(regions))


def region(rid, bbox=(10, 10, 20, 20), manual=False, region_type=RegionType.CLINICAL_SECTION):
    return MedicalRegionData(
        region_id=rid,
        bbox=bbox,
        is_manually_edited=manual,
        region_type=region_type,
    )


def test_unique_geometry_match_is_explicit_and_non_destructive():
    doc = document(region("persistent-r1"))
    before = doc.to_dict()
    result = match_cross_run(
        doc, analysis(hypothesis("hypothesis-r1")), source_id="doc-1"
    )
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.hypothesis_id == "hypothesis-r1"
    assert candidate.persistent_region_id == "persistent-r1"
    assert candidate.state is CrossRunMatchState.MATCHED
    assert doc.to_dict() == before


def test_matching_never_uses_equal_ids():
    doc = document(region("same-id", bbox=(10, 10, 20, 20)))
    result = match_cross_run(
        doc, analysis(hypothesis("same-id", bbox=(60, 60, 20, 20))), source_id="doc-1"
    )
    assert result.candidates == ()
    assert result.unmatched_new == ("same-id",)
    assert result.unmatched_existing == ("same-id",)


def test_ambiguous_candidates_remain_unresolved():
    doc = document(
        region("r1", bbox=(10, 10, 20, 20)),
        region("r2", bbox=(12, 10, 20, 20)),
    )
    result = match_cross_run(
        doc,
        analysis(hypothesis("h1", bbox=(11, 10, 20, 20))),
        source_id="doc-1",
        config=CrossRunMatchingConfig(min_iou=0.8),
    )
    assert {item.state for item in result.candidates} == {CrossRunMatchState.AMBIGUOUS}
    assert result.unmatched_new == ()
    assert result.unmatched_existing == ()


def test_one_to_one_collision_is_ambiguous():
    doc = document(region("r1", bbox=(10, 10, 20, 20)))
    result = match_cross_run(
        doc,
        analysis(
            hypothesis("h1", bbox=(10, 10, 20, 20)),
            hypothesis("h2", bbox=(10, 10, 20, 20)),
        ),
        source_id="doc-1",
    )
    assert all(item.state is CrossRunMatchState.AMBIGUOUS for item in result.candidates)


def test_manual_type_conflict_is_not_auto_accepted():
    doc = document(
        region("r1", manual=True, region_type=RegionType.MEDICAL_TABLE)
    )
    result = match_cross_run(
        doc, analysis(hypothesis("h1")), source_id="doc-1"
    )
    assert result.candidates[0].state is CrossRunMatchState.CONFLICT


def test_unmatched_existing_is_not_a_deletion():
    doc = document(region("r1"))
    result = match_cross_run(
        doc, analysis(hypothesis("h2", bbox=(60, 60, 10, 10))), source_id="doc-1"
    )
    assert result.unmatched_existing == ("r1",)
    assert result.candidates == ()


def test_source_identity_is_explicit():
    doc = document(region("r1"))
    try:
        match_cross_run(doc, analysis(hypothesis("h1"), source_id="other"), source_id="doc-1")
    except ValueError as exc:
        assert "source_id" in str(exc)
    else:
        raise AssertionError("source mismatch must be rejected")


def test_serialization_round_trip():
    doc = document(region("r1"))
    result = match_cross_run(
        doc, analysis(hypothesis("h1")), source_id="doc-1",
        config=CrossRunMatchingConfig(min_iou=0.5),
    )
    assert CrossRunReconciliationResult.from_dict(result.to_dict()) == result


def test_hypothesis_id_is_not_persisted_region_identity():
    doc = document(region("persistent-r1"))
    result = match_cross_run(
        doc, analysis(hypothesis("hypothesis-r1")), source_id="doc-1"
    )
    assert result.candidates[0].persistent_region_id == "persistent-r1"
