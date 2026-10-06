"""Deterministic baseline layout analyzer using grayscale pixel evidence.

This is a reference/baseline detector, not semantic medical layout intelligence.
It detects horizontal bands of sufficiently dark pixels and emits the existing
non-destructive LayoutAnalysisResult contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

from .layout_analysis import AnalysisProvenance, LayoutAnalysisResult, RegionHypothesis
from .ui.models import RegionType


@dataclass(frozen=True)
class ProjectionAnalyzerConfig:
    """Configuration for the deterministic horizontal projection baseline."""

    threshold: float = 200.0
    min_ink_pixels: int = 2
    max_row_gap: int = 1
    min_band_height: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 255.0:
            raise ValueError("threshold must be finite and between 0 and 255")
        if self.min_ink_pixels < 1:
            raise ValueError("min_ink_pixels must be >= 1")
        if self.max_row_gap < 0:
            raise ValueError("max_row_gap must be >= 0")
        if self.min_band_height < 1:
            raise ValueError("min_band_height must be >= 1")


class ProjectionLayoutAnalyzer:
    """Detect horizontal ink bands in source-image pixel space.

    Input is a rectangular grayscale matrix where 0 is black and 255 is white.
    No image library is required. The analyzer never mutates a document.
    """

    name = "projection-baseline"

    def __init__(self, config: ProjectionAnalyzerConfig | None = None) -> None:
        self.config = config or ProjectionAnalyzerConfig()

    def analyze(
        self,
        pixels: Sequence[Sequence[float]],
        provenance: AnalysisProvenance,
    ) -> LayoutAnalysisResult:
        height, width = self._validate_pixels(pixels)
        active_rows: list[bool] = []

        for row in pixels:
            active_rows.append(
                sum(value <= self.config.threshold for value in row)
                >= self.config.min_ink_pixels
            )

        bands: list[tuple[int, int]] = []
        start: int | None = None
        last_active: int | None = None

        for index, active in enumerate(active_rows):
            if active:
                if start is None:
                    start = index
                last_active = index
            elif start is not None and last_active is not None:
                if index - last_active > self.config.max_row_gap + 1:
                    bands.append((start, last_active))
                    start = None
                    last_active = None

        if start is not None and last_active is not None:
            bands.append((start, last_active))

        hypotheses: list[RegionHypothesis] = []
        ordinal = 0
        for top, bottom in bands:
            band_height = bottom - top + 1
            if band_height < self.config.min_band_height:
                continue

            left = width
            right = -1
            for y in range(top, bottom + 1):
                for x, value in enumerate(pixels[y]):
                    if value <= self.config.threshold:
                        left = min(left, x)
                        right = max(right, x)

            if right < left:
                continue

            region_id = self._region_id(provenance.source_id, ordinal)
            hypotheses.append(
                RegionHypothesis(
                    region_id=region_id,
                    region_type=RegionType.CLINICAL_SECTION,
                    bbox=(
                        float(left),
                        float(top),
                        float(right - left + 1),
                        float(band_height),
                    ),
                    source=self.name,
                )
            )
            ordinal += 1

        return LayoutAnalysisResult.from_hypotheses(provenance, hypotheses)

    @staticmethod
    def _region_id(source_id: str, ordinal: int) -> str:
        digest = hashlib.sha256(
            f"{source_id}:projection-band:{ordinal}".encode("utf-8")
        ).hexdigest()[:16]
        return f"projection:{digest}"

    @staticmethod
    def _validate_pixels(pixels: Sequence[Sequence[float]]) -> tuple[int, int]:
        if not pixels:
            raise ValueError("pixels must not be empty")
        height = len(pixels)
        width = len(pixels[0])
        if width == 0:
            raise ValueError("pixel rows must not be empty")

        for row in pixels:
            if len(row) != width:
                raise ValueError("pixels must be rectangular")
            for value in row:
                if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 255.0:
                    raise ValueError("pixel values must be finite and between 0 and 255")
        return height, width
