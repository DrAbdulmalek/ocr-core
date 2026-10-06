from pathlib import Path

import pytest

from ocr_core.ui.models import ConfidenceEvidence, LayoutDocument, MedicalRegionData, RegionType


def test_region_round_trip_preserves_verbatim_text():
    evidence = ConfidenceEvidence("tesseract", 0.91, "engine_native", "ocr_result", "cal-v1")
    region = MedicalRegionData(
        region_type=RegionType.PATIENT_DEMOGRAPHICS,
        bbox=(10, 20, 300, 100),
        confidence_evidence=evidence,
        text_verbatim="ة ى 123",
        normalized_view="ه ي 123",
    )
    payload = region.to_dict()
    assert payload["text_verbatim"] == "ة ى 123"
    assert payload["normalized_view"] == "ه ي 123"
    assert payload["confidence_evidence"]["calibration_version"] == "cal-v1"


def test_negative_dimensions_are_rejected():
    with pytest.raises(ValueError):
        MedicalRegionData(bbox=(0, 0, -1, 10))


def test_layout_document_serializes_without_mutating_source(tmp_path: Path):
    document = LayoutDocument("DOC-1", (2480, 3508), dpi=300)
    original = "ةى"
    document.regions.append(MedicalRegionData(text_verbatim=original))
    target = tmp_path / "layout.json"
    document.save(target)
    assert original == "ةى"
    assert '"text_verbatim": "ةى"' in target.read_text(encoding="utf-8")
