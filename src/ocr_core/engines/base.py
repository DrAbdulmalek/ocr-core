"""Base protocol for ocr-core engines: OCRResult + OCREngine.

Policy: engines that do not produce a real confidence value MUST leave
``confidence`` at 0.0 (never invent a number) — consumers treat 0.0 as
"unknown", mirroring the invented-confidence isolation adopted in the
medical suite (adapter.py P0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OCRResult:
    text: str = ""
    engine: str = ""
    confidence: float = 0.0  # 0.0 = unknown — never a fabricated score
    processing_time: float = 0.0
    error: Optional[str] = None
    pages: int = 1
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


class OCREngine:
    """Abstract engine. Subclasses set ``name`` and implement process_image."""

    name: str = "base"

    def available(self) -> bool:
        return False

    def process_image(self, image_path) -> OCRResult:
        raise NotImplementedError

    def process_pdf(self, pdf_path, max_pages: Optional[int] = None) -> OCRResult:
        raise NotImplementedError