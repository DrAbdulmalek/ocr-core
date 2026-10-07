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
    failure_code: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"complete", "empty", "degraded", "failed"}:
            raise ValueError("status must be complete, empty, degraded, or failed")
        if self.status == "empty" and self.result.hypotheses:
            raise ValueError("empty observation cannot contain hypotheses")
        if self.status == "failed":
            if not self.failure_code or not self.failure_code.strip():
                raise ValueError("failed observation requires failure_code")
            if not self.failure_message or not self.failure_message.strip():
                raise ValueError("failed observation requires failure_message")
        elif self.failure_code is not None or self.failure_message is not None:
            raise ValueError("failure metadata is only valid for failed observations")

    @property
    def provenance(self) -> AnalysisProvenance:
        return self.result.provenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result.to_dict(),
            "quality": [item.to_dict() for item in self.quality],
            "status": self.status,
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalyzerObservation":
        return cls(
            result=LayoutAnalysisResult.from_dict(data["result"]),
            quality=tuple(AnalyzerQuality.from_dict(item) for item in data.get("quality", [])),
            status=str(data.get("status", "complete")),
            failure_code=(str(data["failure_code"]) if data.get("failure_code") is not None else None),
            failure_message=(str(data["failure_message"]) if data.get("failure_message") is not None else None),
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
    failure_code: str | None = None,
    failure_message: str | None = None,
) -> AnalyzerObservation:
    """Validate and wrap an analysis result without mutation."""
    observation = AnalyzerObservation(\n        result=result,\n        quality=quality,\n        status=status,\n        failure_code=failure_code,\n        failure_message=failure_message,\n    )
    validate_analyzer_result(observation)
    return observation
