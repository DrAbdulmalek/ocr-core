from pathlib import Path

import pytest

from ocr_core.ui.models import (
    ConfidenceEvidence,
    LayoutDocument,
    MedicalRegionData,
    RegionType,
)


def test_region_round_trip_preserves_verbatim_text():
    evidence = ConfidenceEvidence(
        "tesseract", 0.91, "engine_native", "ocr_result", "cal-v1"
    )
    region = MedicalRegionData(
        region_type=RegionType.PATIENT_DEMOGRAPHICS,
        bbox=(10, 20, 300, 100),
        confidence_evidence=evidence,
        text_verbatim="ة ى 123",
        normalized_view="ه ي 123",
    )
    payload = region.to_dict()
    restored = MedicalRegionData.from_dict(payload)

    assert restored.text_verbatim == "ة ى 123"
    assert restored.normalized_view == "ه ي 123"
    assert restored.confidence_evidence == evidence
    assert restored.region_type is RegionType.PATIENT_DEMOGRAPHICS


def test_negative_dimensions_are_rejected():
    with pytest.raises(ValueError):
        MedicalRegionData(bbox=(0, 0, -1, 10))


def test_layout_document_serializes_without_mutating_source(tmp_path: Path):
    document = LayoutDocument("DOC-1", (2480, 3508), dpi=300)
    original = "ةى"
    document.regions.append(
        MedicalRegionData(
            region_id="r1",
            bbox=(10, 20, 300, 100),
            text_verbatim=original,
            normalized_view="هى",
        )
    )
    target = tmp_path / "layout.json"

    document.save(target)
    restored = LayoutDocument.load(target)

    assert original == "ةى"
    assert restored.document_id == "DOC-1"
    assert restored.image_dimensions == (2480, 3508)
    assert restored.dpi == 300
    assert restored.regions[0].region_id == "r1"
    assert restored.regions[0].text_verbatim == "ةى"
    assert restored.regions[0].normalized_view == "هى"
