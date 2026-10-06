"""Serializable, non-destructive layout models for the optional region editor."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


class RegionType(str, Enum):
    HEADER_CLINIC = "HEADER_CLINIC"
    PATIENT_DEMOGRAPHICS = "PATIENT_DEMOGRAPHICS"
    CLINICAL_SECTION = "CLINICAL_SECTION"
    MEDICAL_TABLE = "MEDICAL_TABLE"
    HANDWRITTEN_ANNOTATION = "HANDWRITTEN_ANNOTATION"
    FOOTER_SIGNATURE = "FOOTER_SIGNATURE"


@dataclass(frozen=True)
class ConfidenceEvidence:
    """Raw confidence evidence; it is not a cross-engine comparable score."""

    engine: str
    raw_confidence: float
    scale: str
    source: str
    calibration_version: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.raw_confidence <= 1.0:
            raise ValueError("raw_confidence must be between 0 and 1")
        if not self.engine or not self.scale or not self.source:
            raise ValueError("engine, scale and source are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "raw_confidence": self.raw_confidence,
            "scale": self.scale,
            "source": self.source,
            "calibration_version": self.calibration_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConfidenceEvidence":
        return cls(
            engine=str(payload["engine"]),
            raw_confidence=float(payload["raw_confidence"]),
            scale=str(payload["scale"]),
            source=str(payload["source"]),
            calibration_version=payload.get("calibration_version"),
        )


@dataclass
class MedicalRegionData:
    region_id: str = field(default_factory=lambda: str(uuid4()))
    region_type: RegionType | str = RegionType.CLINICAL_SECTION
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    confidence_evidence: ConfidenceEvidence | None = None
    source: str = "user_manual_editor"
    is_manually_edited: bool = True
    text_verbatim: str = ""
    normalized_view: str = ""

    def __post_init__(self) -> None:
        self.region_type = RegionType(self.region_type)
        if len(self.bbox) != 4:
            raise ValueError("bbox must be (x, y, width, height)")
        if any(value < 0 for value in self.bbox[2:]):
            raise ValueError("bbox width and height must be non-negative")
        if not self.region_id:
            raise ValueError("region_id is required")
        if not self.source:
            raise ValueError("source is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "type": self.region_type.value,
            "bbox": list(self.bbox),
            "confidence_evidence": (
                self.confidence_evidence.to_dict()
                if self.confidence_evidence
                else None
            ),
            "source": self.source,
            "is_manually_edited": self.is_manually_edited,
            "text_verbatim": self.text_verbatim,
            "normalized_view": self.normalized_view,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MedicalRegionData":
        evidence = payload.get("confidence_evidence")
        return cls(
            region_id=str(payload["region_id"]),
            region_type=str(payload["type"]),
            bbox=tuple(float(value) for value in payload["bbox"]),
            confidence_evidence=(
                ConfidenceEvidence.from_dict(evidence) if evidence else None
            ),
            source=str(payload.get("source", "unknown")),
            is_manually_edited=bool(payload.get("is_manually_edited", False)),
            text_verbatim=str(payload.get("text_verbatim", "")),
            normalized_view=str(payload.get("normalized_view", "")),
        )


@dataclass
class LayoutDocument:
    document_id: str
    image_dimensions: tuple[int, int]
    dpi: int | None = None
    layout_version: str = "1.0"
    regions: list[MedicalRegionData] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id is required")
        if len(self.image_dimensions) != 2:
            raise ValueError("image_dimensions must be (width, height)")
        if any(value <= 0 for value in self.image_dimensions):
            raise ValueError("image dimensions must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "image_dimensions": {
                "width": self.image_dimensions[0],
                "height": self.image_dimensions[1],
                "dpi": self.dpi,
            },
            "layout_version": self.layout_version,
            "regions": [region.to_dict() for region in self.regions],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LayoutDocument":
        dimensions = payload["image_dimensions"]
        return cls(
            document_id=str(payload["document_id"]),
            image_dimensions=(int(dimensions["width"]), int(dimensions["height"])),
            dpi=dimensions.get("dpi"),
            layout_version=str(payload.get("layout_version", "1.0")),
            regions=[
                MedicalRegionData.from_dict(region)
                for region in payload.get("regions", [])
            ],
        )

    @classmethod
    def from_json(cls, payload: str) -> "LayoutDocument":
        return cls.from_dict(json.loads(payload))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LayoutDocument":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
