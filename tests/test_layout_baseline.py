import pytest

from ocr_core.layout_analysis import AnalysisProvenance, reconcile
from ocr_core.layout_baseline import (
    ProjectionAnalyzerConfig,
    ProjectionLayoutAnalyzer,
)
from ocr_core.ui.models import LayoutDocument


def provenance():
    return AnalysisProvenance("projection-baseline", "1.0", "cfg-a", "image-1")


def test_projection_analyzer_detects_bands_in_pixel_coordinates():
    pixels = [
        [255, 255, 255, 255, 255, 255],
        [255, 20, 20, 255, 255, 255],
        [255, 20, 20, 255, 255, 255],
        [255, 255, 255, 255, 255, 255],
        [255, 255, 255, 255, 255, 255],
        [255, 10, 255, 255, 10, 255],
    ]
    result = ProjectionLayoutAnalyzer(
        ProjectionAnalyzerConfig(min_ink_pixels=2, max_row_gap=0)
    ).analyze(pixels, provenance())

    assert len(result.hypotheses) == 2
    assert result.hypotheses[0].bbox == (1.0, 1.0, 2.0, 2.0)
    assert result.hypotheses[1].bbox == (1.0, 5.0, 4.0, 1.0)


def test_projection_analyzer_is_deterministic():
    pixels = [[255, 10, 10, 255], [255, 255, 255, 255]]
    analyzer = ProjectionLayoutAnalyzer()
    first = analyzer.analyze(pixels, provenance())
    second = analyzer.analyze(pixels, provenance())
    assert first == second


def test_projection_analyzer_merges_small_row_gaps():
    pixels = [
        [10, 10, 255],
        [255, 255, 255],
        [10, 10, 255],
    ]
    result = ProjectionLayoutAnalyzer(
        ProjectionAnalyzerConfig(max_row_gap=1)
    ).analyze(pixels, provenance())
    assert len(result.hypotheses) == 1
    assert result.hypotheses[0].bbox == (0.0, 0.0, 2.0, 3.0)


def test_projection_analyzer_empty_or_non_rectangular_input_is_rejected():
    analyzer = ProjectionLayoutAnalyzer()
    with pytest.raises(ValueError):
        analyzer.analyze([], provenance())
    with pytest.raises(ValueError):
        analyzer.analyze([[0], [0, 0]], provenance())


def test_projection_analyzer_rejects_invalid_pixels():
    analyzer = ProjectionLayoutAnalyzer()
    with pytest.raises(ValueError):
        analyzer.analyze([[256]], provenance())
    with pytest.raises(ValueError):
        analyzer.analyze([[float("nan")]], provenance())


def test_projection_result_reconciles_without_mutating_document():
    pixels = [[255, 10, 10, 255]]
    result = ProjectionLayoutAnalyzer().analyze(pixels, provenance())
    document = LayoutDocument("doc-1", (4, 1))
    reconciliation = reconcile(document, result)

    assert len(reconciliation.items) == 1
    assert reconciliation.items[0].region_id == result.hypotheses[0].region_id
    assert document.regions == []
