"""Production layout analyzer boundary primitives.

This module defines a dependency-free boundary around analyzer observations.
It deliberately does not mutate LayoutDocument or perform matching/apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from .layout_analysis import AnalysisProvenance, LayoutAnalysisResult, RegionHypothesis


@dataclass(frozen=True)
class AnalyzerQuality:
    """Analyzer-specific observation quality; never OCR confidence."""

    metric: str
    value: float
    scale: str
    source: str
    version: str

    def __post_init__(self) -> None:
        if not self.metric.strip() or not self.scale.strip() or not self.source.strip() or not self.version.strip():
            raise ValueError("quality metadata fields must be non-empty")
        if not isfinite(self.value):
            raise ValueError("quality value must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "scale": self.scale,
            "source": self.source,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalyzerQuality":
        return cls(
            metric=str(data["metric"]),
            value=float(data["value"]),
            scale=str(data["scale"]),
            source=str(data["source"]),
            version=str(data["version"]),
        )


@dataclass(frozen=True)
class AnalyzerObservation:
    """Immutable analyzer output envelope.

    The canonical analysis result remains the existing LayoutAnalysisResult.
    """

    result: LayoutAnalysisResult
    quality: tuple[AnalyzerQuality, ...] = ()
    status: str = "complete"

    def __post_init__(self) -> None:
        if self.status not in {"complete", "empty", "degraded"}:
            raise ValueError("status must be complete, empty, or degraded")
        if self.status == "empty" and self.result.hypotheses:
            raise ValueError("empty observation cannot contain hypotheses")

    @property
    def provenance(self) -> AnalysisProvenance:
        return self.result.provenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result.to_dict(),
            "quality": [item.to_dict() for item in self.quality],
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalyzerObservation":
        return cls(
            result=LayoutAnalysisResult.from_dict(data["result"]),
            quality=tuple(AnalyzerQuality.from_dict(item) for item in data.get("quality", [])),
            status=str(data.get("status", "complete")),
        )


def validate_analyzer_result(observation: AnalyzerObservation) -> None:
    """Validate analyzer output without touching any canonical document."""
    result = observation.result
    if not result.provenance.source_id.strip():
        raise ValueError("source_id must be non-empty")
    if not result.provenance.analyzer.strip():
        raise ValueError("analyzer must be non-empty")
    if not result.provenance.analyzer_version.strip():
        raise ValueError("analyzer_version must be non-empty")
    if not result.provenance.configuration_id.strip():
        raise ValueError("configuration_id must be non-empty")
    for hypothesis in result.hypotheses:
        x, y, width, height = hypothesis.bbox
        if not all(isfinite(value) for value in (x, y, width, height)):
            raise ValueError("hypothesis bbox must be finite")
        if width < 0 or height < 0:
            raise ValueError("hypothesis bbox dimensions must be non-negative")


def make_observation(
    result: LayoutAnalysisResult,
    *,
    quality: tuple[AnalyzerQuality, ...] = (),
    status: str = "complete",
) -> AnalyzerObservation:
    """Validate and wrap an analysis result without mutation."""
    observation = AnalyzerObservation(result=result, quality=quality, status=status)
    validate_analyzer_result(observation)
    return observation
