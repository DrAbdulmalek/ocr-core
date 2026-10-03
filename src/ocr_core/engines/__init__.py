"""Engines subpackage — base protocol only imported here (optional deps stay lazy)."""
from ocr_core.engines.base import OCRResult, OCREngine  # noqa: F401

__all__ = ["OCRResult", "OCREngine"]