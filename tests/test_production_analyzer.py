from ocr_core.layout_analysis import AnalysisProvenance, LayoutAnalysisResult, RegionHypothesis
from ocr_core.production_analyzer import AnalyzerQuality, make_observation, validate_analyzer_result
from ocr_core.ui.models import RegionType


def _result(hypotheses=()):
    return LayoutAnalysisResult.from_hypotheses(
        AnalysisProvenance(
            analyzer="baseline",
            analyzer_version="1.0",
            configuration_id="cfg-a",
            source_id="doc-1",
            analysis_run_id="run-1",
        ),
        hypotheses,
    )


def test_complete_observation_round_trip_is_deterministic():
    result = _result(
        [
            RegionHypothesis("h-2", RegionType.CLINICAL_SECTION, (10, 20, 30, 40)),
            RegionHypothesis("h-1", RegionType.HEADER_CLINIC, (1, 2, 3, 4)),
        ]
    )
    observation = make_observation(
        result,
        quality=(AnalyzerQuality("geometry_quality", 0.9, "0..1", "baseline", "1"),),
    )

    restored = observation.from_dict(observation.to_dict())

    assert restored == observation
    assert list(restored.result.hypotheses)[0].region_id == "h-1"


def test_empty_result_is_valid_and_has_no_deletion_semantics():
    observation = make_observation(_result(), status="empty")
    assert observation.status == "empty"
    assert observation.result.hypotheses == ()


def test_degraded_result_remains_observation_only():
    observation = make_observation(_result(), status="degraded")
    assert observation.status == "degraded"


def test_quality_is_separate_from_ocr_confidence():
    quality = AnalyzerQuality("detector_score", 0.8, "0..1", "model-x", "2")
    assert quality.to_dict()["metric"] == "detector_score"
    assert not hasattr(quality, "raw_confidence")


def test_validation_rejects_invalid_quality_and_status():
    try:
        AnalyzerQuality("q", float("nan"), "0..1", "x", "1")
        assert False
    except ValueError:
        pass

    try:
        make_observation(_result(), status="failed")
        assert False
    except ValueError:
        pass


def test_analysis_provenance_is_explicit():
    observation = make_observation(_result())
    assert observation.provenance.analyzer == "baseline"
    assert observation.provenance.configuration_id == "cfg-a"
    assert observation.provenance.analysis_run_id == "run-1"


def test_validation_does_not_mutate_any_document():
    observation = make_observation(
        _result([RegionHypothesis("h", RegionType.CLINICAL_SECTION, (0, 0, 10, 10))])
    )
    before = observation.to_dict()
    validate_analyzer_result(observation)
    assert observation.to_dict() == before


def test_failed_observation_requires_explicit_failure_metadata_and_round_trips():
    observation = make_observation(
        _result(),
        status="failed",
        failure_code="INPUT_INVALID",
        failure_message="pixel evidence is not rectangular",
    )
    restored = observation.from_dict(observation.to_dict())
    assert restored == observation
    assert restored.failure_code == "INPUT_INVALID"


def test_failed_observation_rejects_missing_failure_metadata():
    try:
        make_observation(_result(), status="failed")
        assert False
    except ValueError:
        pass


def test_baseline_analyzer_output_fits_production_boundary():
    from ocr_core.layout_baseline import ProjectionAnalyzerConfig, ProjectionLayoutAnalyzer

    pixels = [[255, 10, 10, 255], [255, 255, 255, 255]]
    result = ProjectionLayoutAnalyzer(
        ProjectionAnalyzerConfig(min_ink_pixels=2)
    ).analyze(pixels, _result().provenance)
    observation = make_observation(result)

    assert observation.status == "complete"
    assert observation.provenance.analyzer == "baseline"
    assert observation.result.hypotheses[0].bbox == (1.0, 0.0, 2.0, 1.0)
