"""Tests for selective (regional) re-recognition — clean-room module."""
from __future__ import annotations

from pathlib import Path

import pytest

from ocr_core.engines.base import OCREngine, OCRResult
from ocr_core.regional import (
    extract_region,
    normalize_bbox,
    rerun_region,
    rerun_regions,
)


class RegionEngine(OCREngine):
    name = "region-fake"

    def __init__(self, available: bool = True):
        self._available = available
        self.last_path: Path | None = None

    def available(self) -> bool:
        return self._available

    def process_image(self, image_path) -> OCRResult:
        self.last_path = Path(image_path)
        return OCRResult(text="region text", engine=self.name, confidence=0.0)

    def process_pdf(self, pdf_path, max_pages=None) -> OCRResult:  # pragma: no cover
        return OCRResult(error="pdf unsupported in fake")


def _png(tmp_path: Path, size=(100, 80)) -> Path:
    from PIL import Image

    p = tmp_path / "src.png"
    Image.new("RGB", size, "white").save(p)
    return p


# ----------------------------------------------------------------------
# normalize_bbox
# ----------------------------------------------------------------------
def test_normalize_bbox_xywh_passthrough():
    assert normalize_bbox((10, 20, 30, 40)) == (10, 20, 30, 40)


def test_normalize_bbox_xyxy_converts_to_xywh():
    assert normalize_bbox((10, 20, 40, 60), bbox_format="xyxy") == (10, 20, 30, 40)


def test_normalize_bbox_rejects_bad_shapes():
    with pytest.raises(ValueError):
        normalize_bbox((1, 2, 3))
    with pytest.raises(ValueError):
        normalize_bbox((1, 2, 3, 4), bbox_format="corners")
    with pytest.raises(ValueError):
        normalize_bbox((1, 2, 0, 4))  # non-positive width
    with pytest.raises(ValueError):
        normalize_bbox((5, 5, 5, 5), bbox_format="xyxy")  # x2 == x1
    with pytest.raises(ValueError):
        normalize_bbox(("a", 2, 3, 4))


# ----------------------------------------------------------------------
# extract_region
# ----------------------------------------------------------------------
def test_extract_region_crops_and_returns_png(tmp_path):
    src = _png(tmp_path)
    out = extract_region(src, (10, 10, 30, 20))
    assert out.exists() and out.suffix == ".png"
    from PIL import Image

    with Image.open(out) as img:
        assert img.size == (30, 20)
    out.unlink()


def test_extract_region_padding_grows_box_within_bounds(tmp_path):
    src = _png(tmp_path, (50, 50))
    out = extract_region(src, (10, 10, 10, 10), padding=5)
    from PIL import Image

    with Image.open(out) as img:
        assert img.size == (20, 20)  # 10+2*5, clipped inside 50x50
    out.unlink()


def test_extract_region_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_region(tmp_path / "nope.png", (0, 0, 5, 5))


def test_extract_region_zero_after_clip_raises(tmp_path):
    with pytest.raises(ValueError):
        extract_region(_png(tmp_path, (10, 10)), (20, 20, 5, 5))


# ----------------------------------------------------------------------
# rerun_region
# ----------------------------------------------------------------------
def test_rerun_region_happy_path(tmp_path):
    engine = RegionEngine()
    result = rerun_region(_png(tmp_path), (5, 5, 40, 30), engine=engine)
    assert result.ok and result.text == "region text" and result.engine == "region-fake"
    assert result.confidence == 0.0  # policy: fake engines report unknown
    assert engine.last_path is not None and engine.last_path.exists() is False


def test_rerun_region_xyxy_format(tmp_path):
    engine = RegionEngine()
    result = rerun_region(_png(tmp_path), (5, 5, 45, 35), engine=engine, bbox_format="xyxy")
    assert result.ok


def test_rerun_region_no_engine_available(tmp_path):
    result = rerun_region(_png(tmp_path), (0, 0, 10, 10), engine=RegionEngine(available=False))
    assert result.error is not None and result.text == ""


def test_rerun_region_bad_bbox_reported_not_raised(tmp_path):
    result = rerun_region(_png(tmp_path), (0, 0, 0, 0), engine=RegionEngine())
    assert result.error is not None and "invalid bbox" in result.error


def test_rerun_region_missing_source_reported(tmp_path):
    result = rerun_region(tmp_path / "gone.png", (0, 0, 10, 10), engine=RegionEngine())
    assert result.error is not None and "not found" in result.error


# ----------------------------------------------------------------------
# rerun_regions
# ----------------------------------------------------------------------
def test_rerun_regions_batch_order_preserved(tmp_path):
    engine = RegionEngine()
    boxes = [(0, 0, 20, 20), (30, 0, 20, 20), (0, 40, 20, 20)]
    results = rerun_regions(_png(tmp_path), boxes, engine=engine)
    assert len(results) == 3
    assert all(r.ok for r in results)


def test_rerun_regions_with_unavailable_engine_all_report_error(tmp_path):
    results = rerun_regions(_png(tmp_path), [(0, 0, 10, 10), (5, 5, 10, 10)], engine=RegionEngine(available=False))
    assert len(results) == 2
    assert all(r.error for r in results)
