"""Selective (regional) re-recognition — clean-room design.

Concept provenance (see ``docs/research/abbyy_features.md``, decision D-004):
commercial document tools expose "select a region and re-run recognition on
it". This module implements that concept from scratch on top of ocr-core
engines — no proprietary code or models are involved.

Coordinate conventions
----------------------
``bbox_format`` accepts two shapes:

- ``"xywh"`` (default) — ``(x, y, width, height)``; matches
  ``ocr_core.ui.models.MedicalRegionData`` and
  ``ocr_core.layout_baseline`` pixel-space boxes.
- ``"xyxy"`` — ``(x1, y1, x2, y2)``; matches bounding-box editors that
  store corner coordinates (e.g. training-snippet UIs).

Policy: failures are reported through ``OCRResult.error`` and never raised,
mirroring ``ocr_core.engines.base``.
"""
from __future__ import annotations

import logging
import math
import tempfile
from collections.abc import Sequence
from pathlib import Path

from ocr_core.engines.base import OCREngine, OCRResult

logger = logging.getLogger(__name__)

__all__ = [
    "extract_region",
    "normalize_bbox",
    "rerun_region",
    "rerun_regions",
]


def normalize_bbox(
    bbox: Sequence[float],
    bbox_format: str = "xywh",
) -> tuple[int, int, int, int]:
    """Return an ``(x, y, w, h)`` tuple of non-negative ints.

    Raises ``ValueError`` for malformed boxes (bad length, negative
    coordinates, non-positive width/height in xywh, x2<=x1 or y2<=y1 in
    xyxy) — callers decide how to surface the failure.
    """
    if bbox_format not in ("xywh", "xyxy"):
        raise ValueError(f"bbox_format must be 'xywh' or 'xyxy', got {bbox_format!r}")
    if len(bbox) != 4:
        raise ValueError("bbox must have exactly 4 values")
    try:
        values = tuple(float(v) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bbox values must be numeric: {exc}") from exc
    if any(math.isnan(v) for v in values):  # NaN guard
        raise ValueError("bbox values must not be NaN")
    if bbox_format == "xywh":
        x, y, w, h = values
        if x < 0 or y < 0:
            raise ValueError("xywh bbox origin must be non-negative")
        if w <= 0 or h <= 0:
            raise ValueError("xywh bbox width/height must be positive")
        return int(round(x)), int(round(y)), int(round(w)), int(round(h))
    x1, y1, x2, y2 = values
    if x1 < 0 or y1 < 0:
        raise ValueError("xyxy bbox origin must be non-negative")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("xyxy bbox requires x2 > x1 and y2 > y1")
    return int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))


def _clip_bbox(bbox: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Clip an xywh box to image bounds; returns a possibly smaller box."""
    x, y, w, h = bbox
    width, height = size
    x = max(0, min(int(x), width))
    y = max(0, min(int(y), height))
    w = max(0, min(int(w), width - x))
    h = max(0, min(int(h), height - y))
    return x, y, w, h


def extract_region(
    image_path: str | Path,
    bbox: Sequence[float],
    bbox_format: str = "xywh",
    padding: int = 0,
) -> Path:
    """Crop ``bbox`` out of ``image_path`` into a temp PNG and return its path.

    Uses Pillow (available with any engine extra). Raises ``FileNotFoundError``
    for missing sources and ``ValueError`` for malformed boxes — pure data
    errors, distinct from OCR failures which use ``OCRResult.error``.
    """
    from PIL import Image  # engine extras guarantee pillow

    src = Path(image_path)
    if not src.exists():
        raise FileNotFoundError(f"image not found: {src}")
    x, y, w, h = normalize_bbox(bbox, bbox_format)
    with Image.open(src) as img:
        width, height = img.size
        if padding:
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(width - x, w + 2 * padding)
            h = min(height - y, h + 2 * padding)
        x, y, w, h = _clip_bbox((x, y, w, h), (width, height))
        if w == 0 or h == 0:
            raise ValueError("region is empty after clipping to image bounds")
        crop = img.crop((x, y, x + w, y + h))
        tmp = tempfile.NamedTemporaryFile(prefix="ocr_core_region_", suffix=".png", delete=False)
        tmp.close()
        crop.save(tmp.name, format="PNG")
    return Path(tmp.name)


def _resolve_engine(engine: OCREngine | None) -> OCREngine | None:
    if engine is not None:
        return engine if engine.available() else None
    try:
        from ocr_core.pipeline import _resolve_engine as _resolver

        return _resolver(None)
    except Exception:
        return None


def rerun_region(
    image_path: str | Path,
    bbox: Sequence[float],
    engine: OCREngine | None = None,
    bbox_format: str = "xywh",
    padding: int = 0,
) -> OCRResult:
    """Recognize only ``bbox`` from ``image_path`` with the given engine.

    Never raises for OCR problems: missing engines, missing files and bad
    boxes are all reported through ``OCRResult.error``.
    """
    chosen = _resolve_engine(engine)
    if chosen is None:
        return OCRResult(engine="", error="no OCR engine available for regional re-recognition")
    region_path: Path | None = None
    try:
        region_path = extract_region(image_path, bbox, bbox_format=bbox_format, padding=padding)
    except FileNotFoundError as exc:
        return OCRResult(engine="", error=str(exc))
    except ValueError as exc:
        return OCRResult(engine="", error=f"invalid bbox {tuple(bbox)} ({bbox_format}): {exc}")
    try:
        return chosen.process_image(region_path)
    except Exception as exc:  # engine contract says don't raise, but stay safe
        logger.debug("regional recognition crashed", exc_info=True)
        return OCRResult(engine=chosen.name, error=f"engine crash during regional OCR: {exc}")
    finally:
        if region_path is not None:
            try:
                region_path.unlink(missing_ok=True)
            except Exception:
                pass


def rerun_regions(
    image_path: str | Path,
    bboxes: Sequence[Sequence[float]],
    engine: OCREngine | None = None,
    bbox_format: str = "xywh",
    padding: int = 0,
) -> list[OCRResult]:
    """Batch form of :func:`rerun_region` (one result per bbox, same order)."""
    return [
        rerun_region(image_path, bbox, engine=engine, bbox_format=bbox_format, padding=padding)
        for bbox in bboxes
    ]
